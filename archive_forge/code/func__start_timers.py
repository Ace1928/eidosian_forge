import logging
import socket
import struct
import traceback
from socket import IPPROTO_TCP, TCP_NODELAY
from eventlet import semaphore
from os_ken.lib.packet import bgp
from os_ken.lib.packet.bgp import AS_TRANS
from os_ken.lib.packet.bgp import BGPMessage
from os_ken.lib.packet.bgp import BGPOpen
from os_ken.lib.packet.bgp import BGPUpdate
from os_ken.lib.packet.bgp import BGPKeepAlive
from os_ken.lib.packet.bgp import BGPNotification
from os_ken.lib.packet.bgp import BGP_MSG_OPEN
from os_ken.lib.packet.bgp import BGP_MSG_UPDATE
from os_ken.lib.packet.bgp import BGP_MSG_KEEPALIVE
from os_ken.lib.packet.bgp import BGP_MSG_NOTIFICATION
from os_ken.lib.packet.bgp import BGP_MSG_ROUTE_REFRESH
from os_ken.lib.packet.bgp import BGP_CAP_FOUR_OCTET_AS_NUMBER
from os_ken.lib.packet.bgp import BGP_CAP_ENHANCED_ROUTE_REFRESH
from os_ken.lib.packet.bgp import BGP_CAP_MULTIPROTOCOL
from os_ken.lib.packet.bgp import BGP_ERROR_HOLD_TIMER_EXPIRED
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_HOLD_TIMER_EXPIRED
from os_ken.lib.packet.bgp import get_rf
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import CORE_ERROR_CODE
from os_ken.services.protocols.bgp.constants import BGP_FSM_CONNECT
from os_ken.services.protocols.bgp.constants import BGP_FSM_OPEN_CONFIRM
from os_ken.services.protocols.bgp.constants import BGP_FSM_OPEN_SENT
from os_ken.services.protocols.bgp.constants import BGP_VERSION_NUM
from os_ken.services.protocols.bgp.protocol import Protocol
def _start_timers(self, peer_holdtime):
    """Starts keepalive and expire timers.

        Hold time is set to min. of peer and configured/default hold time.
        Starts keep alive timer and expire timer based on this value.
        """
    neg_timer = min(self._holdtime, peer_holdtime)
    if neg_timer < self._holdtime:
        LOG.info('Negotiated hold time (%s) is lower then configured/default (%s).', neg_timer, self._holdtime)
    self._holdtime = neg_timer
    self._keepalive = self._create_timer('Keepalive Timer', self._send_keepalive)
    interval = self._holdtime // 3
    self._keepalive.start(interval, now=False)
    self._expiry = self._create_timer('Holdtime Timer', self._expired)
    self._expiry.start(self._holdtime, now=False)
    LOG.debug('Started keep-alive and expire timer for negotiated holdtime %s', self._holdtime)