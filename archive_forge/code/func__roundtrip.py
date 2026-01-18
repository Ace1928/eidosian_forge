from .err import OperationalError
from functools import partial
import hashlib
def _roundtrip(conn, send_data):
    conn.write_packet(send_data)
    pkt = conn._read_packet()
    pkt.check_error()
    return pkt