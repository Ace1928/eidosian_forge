import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
def _handle_receive_pack_tail(self, proto: Protocol, capabilities: Set[bytes], progress: Optional[Callable[[bytes], None]]=None) -> Optional[Dict[bytes, Optional[str]]]:
    """Handle the tail of a 'git-receive-pack' request.

        Args:
          proto: Protocol object to read from
          capabilities: List of negotiated capabilities
          progress: Optional progress reporting function

        Returns:
          dict mapping ref name to:
            error message if the ref failed to update
            None if it was updated successfully
        """
    if CAPABILITY_SIDE_BAND_64K in capabilities:
        if progress is None:

            def progress(x):
                pass
        if CAPABILITY_REPORT_STATUS in capabilities:
            assert self._report_status_parser is not None
            pktline_parser = PktLineParser(self._report_status_parser.handle_packet)
        for chan, data in _read_side_band64k_data(proto.read_pkt_seq()):
            if chan == SIDE_BAND_CHANNEL_DATA:
                if CAPABILITY_REPORT_STATUS in capabilities:
                    pktline_parser.parse(data)
            elif chan == SIDE_BAND_CHANNEL_PROGRESS:
                progress(data)
            else:
                raise AssertionError('Invalid sideband channel %d' % chan)
    elif CAPABILITY_REPORT_STATUS in capabilities:
        assert self._report_status_parser
        for pkt in proto.read_pkt_seq():
            self._report_status_parser.handle_packet(pkt)
    if self._report_status_parser is not None:
        return dict(self._report_status_parser.check())
    return None