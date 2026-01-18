import zlib
from typing import Callable, List, Optional, Tuple
from zlib import Z_FINISH, Z_SYNC_FLUSH
Write some bytes to the chunk.

        If the bytes fit, False is returned. Otherwise True is returned
        and the bytes have not been added to the chunk.

        :param bytes: The bytes to include
        :param reserved: If True, we can use the space reserved in the
            constructor.
        