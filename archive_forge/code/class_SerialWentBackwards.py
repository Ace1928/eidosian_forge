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
class SerialWentBackwards(dns.exception.FormError):
    """The current serial number is less than the serial we know."""