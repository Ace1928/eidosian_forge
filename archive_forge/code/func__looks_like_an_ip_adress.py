from __future__ import annotations
from typing import Final
from streamlit import util
from streamlit.logger import get_logger
def _looks_like_an_ip_adress(address: str | None) -> bool:
    if address is None:
        return False
    import socket
    try:
        socket.inet_pton(socket.AF_INET, address)
        return True
    except (AttributeError, OSError):
        pass
    try:
        socket.inet_pton(socket.AF_INET6, address)
        return True
    except (AttributeError, OSError):
        pass
    return False