from ..base import AsyncBase, AsyncIndirectBase
from .utils import delegate_to_executor, proxy_method_directly, proxy_property_directly
@delegate_to_executor('close', 'flush', 'isatty', 'read', 'readall', 'readinto', 'readline', 'readlines', 'seek', 'seekable', 'tell', 'truncate', 'writable', 'write', 'writelines')
@proxy_method_directly('fileno', 'readable')
@proxy_property_directly('closed', 'name', 'mode')
class AsyncIndirectFileIO(AsyncIndirectBase):
    """The indirect asyncio executor version of io.FileIO."""