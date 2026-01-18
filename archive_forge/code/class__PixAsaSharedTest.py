from __future__ import absolute_import, division, print_function
import logging
from passlib import hash, exc
from passlib.utils.compat import u
from .utils import UserHandlerMixin, HandlerCase, repeat_string
from .test_handlers import UPASS_TABLE
class _PixAsaSharedTest(UserHandlerMixin, HandlerCase):
    """
    class w/ shared info for PIX & ASA tests.
    """
    __unittest_skip = True
    requires_user = False
    pix_asa_shared_hashes = [(('cisco', ''), '2KFQnbNIdI.2KYOU'), (('hsc', ''), 'YtT8/k6Np8F1yz2c'), (('', ''), '8Ry2YjIyt7RRXU24'), (('cisco', 'john'), 'hN7LzeyYjw12FSIU'), (('cisco', 'jack'), '7DrfeZ7cyOj/PslD'), (('ripper', 'alex'), 'h3mJrcH0901pqX/m'), (('cisco', 'cisco'), '3USUcOPFUiMCO4Jk'), (('cisco', 'cisco1'), '3USUcOPFUiMCO4Jk'), (('CscFw-ITC!', 'admcom'), 'lZt7HSIXw3.QP7.R'), ('cangetin', 'TynyB./ftknE77QP'), (('cangetin', 'rramsey'), 'jgBZqYtsWfGcUKDi'), (('phonehome', 'rharris'), 'zyIIMSYjiPm0L7a6'), (('cangetin', ''), 'TynyB./ftknE77QP'), (('cangetin', 'rramsey'), 'jgBZqYtsWfGcUKDi'), ('test1', 'TRPEas6f/aa6JSPL'), ('test2', 'OMT6mXmAvGyzrCtp'), ('test3', 'gTC7RIy1XJzagmLm'), ('test4', 'oWC1WRwqlBlbpf/O'), ('password', 'NuLKvvWGg.x9HEKO'), ('0123456789abcdef', '.7nfVBEIEu4KbF/1'), (('1234567890123456', ''), 'feCkwUGktTCAgIbD'), (('watag00s1am', ''), 'jMorNbK0514fadBh'), (('cisco1', 'cisco1'), 'jmINXNH6p1BxUppp'), (UPASS_TABLE, 'CaiIvkLMu2TOHXGT'), (('1234', ''), 'RLPMUQ26KL4blgFN'), (('01234567', ''), '0T52THgnYdV1tlOF'), (('01234567', '3'), '.z0dT9Alkdc7EIGS'), (('01234567', '36'), 'CC3Lam53t/mHhoE7'), (('01234567', '365'), '8xPrWpNnBdD2DzdZ'), (('01234567', '3333'), '.z0dT9Alkdc7EIGS'), (('01234567', '3636'), 'CC3Lam53t/mHhoE7'), (('01234567', '3653'), '8xPrWpNnBdD2DzdZ'), (('01234567', 'adm'), 'dfWs2qiao6KD/P2L'), (('01234567', 'adma'), 'dfWs2qiao6KD/P2L'), (('01234567', 'admad'), 'dfWs2qiao6KD/P2L'), (('01234567', 'user'), 'PNZ4ycbbZ0jp1.j1'), (('01234567', 'user1234'), 'PNZ4ycbbZ0jp1.j1'), (('0123456789ab', ''), 'S31BxZOGlAigndcJ'), (('0123456789ab', '36'), 'wFqSX91X5.YaRKsi'), (('0123456789ab', '365'), 'qjgo3kNgTVxExbno'), (('0123456789ab', '3333'), 'mcXPL/vIZcIxLUQs'), (('0123456789ab', '3636'), 'wFqSX91X5.YaRKsi'), (('0123456789ab', '3653'), 'qjgo3kNgTVxExbno'), (('0123456789ab', 'user'), 'f.T4BKdzdNkjxQl7'), (('0123456789ab', 'user1234'), 'f.T4BKdzdNkjxQl7'), ((u('táble').encode('utf-8'), 'user'), 'Og8fB4NyF0m5Ed9c'), ((u('táble').encode('utf-8').decode('latin-1').encode('utf-8'), 'user'), 'cMvFC2XVBmK/68yB')]

    def test_calc_digest_spoiler(self):
        """
        _calc_checksum() -- spoil oversize passwords during verify

        for details, see 'spoil_digest' flag instead that function.
        this helps cisco_pix/cisco_asa implement their policy of
        ``.truncate_verify_reject=True``.
        """

        def calc(secret, for_hash=False):
            return self.handler(use_defaults=for_hash)._calc_checksum(secret)
        short_secret = repeat_string('1234', self.handler.truncate_size)
        short_hash = calc(short_secret)
        long_secret = short_secret + 'X'
        long_hash = calc(long_secret)
        self.assertNotEqual(long_hash, short_hash)
        alt_long_secret = short_secret + 'Y'
        alt_long_hash = calc(alt_long_secret)
        self.assertNotEqual(alt_long_hash, short_hash)
        self.assertNotEqual(alt_long_hash, long_hash)
        calc(short_secret, for_hash=True)
        self.assertRaises(exc.PasswordSizeError, calc, long_secret, for_hash=True)
        self.assertRaises(exc.PasswordSizeError, calc, alt_long_secret, for_hash=True)