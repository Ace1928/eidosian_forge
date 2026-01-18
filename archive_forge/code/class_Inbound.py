from typing import Any, List, Optional, Tuple, Union
import dns.exception
import dns.message
import dns.name
import dns.rcode
import dns.rdataset
import dns.rdatatype
import dns.serial
import dns.transaction
import dns.tsig
import dns.zone
class Inbound:
    """
    State machine for zone transfers.
    """

    def __init__(self, txn_manager: dns.transaction.TransactionManager, rdtype: dns.rdatatype.RdataType=dns.rdatatype.AXFR, serial: Optional[int]=None, is_udp: bool=False):
        """Initialize an inbound zone transfer.

        *txn_manager* is a :py:class:`dns.transaction.TransactionManager`.

        *rdtype* can be `dns.rdatatype.AXFR` or `dns.rdatatype.IXFR`

        *serial* is the base serial number for IXFRs, and is required in
        that case.

        *is_udp*, a ``bool`` indidicates if UDP is being used for this
        XFR.
        """
        self.txn_manager = txn_manager
        self.txn: Optional[dns.transaction.Transaction] = None
        self.rdtype = rdtype
        if rdtype == dns.rdatatype.IXFR:
            if serial is None:
                raise ValueError('a starting serial must be supplied for IXFRs')
        elif is_udp:
            raise ValueError('is_udp specified for AXFR')
        self.serial = serial
        self.is_udp = is_udp
        _, _, self.origin = txn_manager.origin_information()
        self.soa_rdataset: Optional[dns.rdataset.Rdataset] = None
        self.done = False
        self.expecting_SOA = False
        self.delete_mode = False

    def process_message(self, message: dns.message.Message) -> bool:
        """Process one message in the transfer.

        The message should have the same relativization as was specified when
        the `dns.xfr.Inbound` was created.  The message should also have been
        created with `one_rr_per_rrset=True` because order matters.

        Returns `True` if the transfer is complete, and `False` otherwise.
        """
        if self.txn is None:
            replacement = self.rdtype == dns.rdatatype.AXFR
            self.txn = self.txn_manager.writer(replacement)
        rcode = message.rcode()
        if rcode != dns.rcode.NOERROR:
            raise TransferError(rcode)
        if len(message.question) > 0:
            if message.question[0].name != self.origin:
                raise dns.exception.FormError('wrong question name')
            if message.question[0].rdtype != self.rdtype:
                raise dns.exception.FormError('wrong question rdatatype')
        answer_index = 0
        if self.soa_rdataset is None:
            if not message.answer or message.answer[0].name != self.origin:
                raise dns.exception.FormError('No answer or RRset not for zone origin')
            rrset = message.answer[0]
            rdataset = rrset
            if rdataset.rdtype != dns.rdatatype.SOA:
                raise dns.exception.FormError('first RRset is not an SOA')
            answer_index = 1
            self.soa_rdataset = rdataset.copy()
            if self.rdtype == dns.rdatatype.IXFR:
                if self.soa_rdataset[0].serial == self.serial:
                    self.done = True
                elif dns.serial.Serial(self.soa_rdataset[0].serial) < self.serial:
                    raise SerialWentBackwards
                else:
                    if self.is_udp and len(message.answer[answer_index:]) == 0:
                        raise UseTCP
                    self.expecting_SOA = True
        for rrset in message.answer[answer_index:]:
            name = rrset.name
            rdataset = rrset
            if self.done:
                raise dns.exception.FormError('answers after final SOA')
            assert self.txn is not None
            if rdataset.rdtype == dns.rdatatype.SOA and name == self.origin:
                if self.rdtype == dns.rdatatype.IXFR:
                    self.delete_mode = not self.delete_mode
                if rdataset == self.soa_rdataset and (self.rdtype == dns.rdatatype.AXFR or (self.rdtype == dns.rdatatype.IXFR and self.delete_mode)):
                    if self.expecting_SOA:
                        raise dns.exception.FormError('empty IXFR sequence')
                    if self.rdtype == dns.rdatatype.IXFR and self.serial != rdataset[0].serial:
                        raise dns.exception.FormError('unexpected end of IXFR sequence')
                    self.txn.replace(name, rdataset)
                    self.txn.commit()
                    self.txn = None
                    self.done = True
                else:
                    self.expecting_SOA = False
                    if self.rdtype == dns.rdatatype.IXFR:
                        if self.delete_mode:
                            if rdataset[0].serial != self.serial:
                                raise dns.exception.FormError('IXFR base serial mismatch')
                        else:
                            self.serial = rdataset[0].serial
                            self.txn.replace(name, rdataset)
                    else:
                        raise dns.exception.FormError('unexpected origin SOA in AXFR')
                continue
            if self.expecting_SOA:
                self.rdtype = dns.rdatatype.AXFR
                self.expecting_SOA = False
                self.delete_mode = False
                self.txn.rollback()
                self.txn = self.txn_manager.writer(True)
            if self.delete_mode:
                self.txn.delete_exact(name, rdataset)
            else:
                self.txn.add(name, rdataset)
        if self.is_udp and (not self.done):
            raise dns.exception.FormError('unexpected end of UDP IXFR')
        return self.done

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.txn:
            self.txn.rollback()
        return False