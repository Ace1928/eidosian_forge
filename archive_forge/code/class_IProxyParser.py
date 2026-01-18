from typing import Tuple, Union
import zope.interface
class IProxyParser(zope.interface.Interface):
    """
    Streaming parser that handles PROXY protocol headers.
    """

    def feed(data: bytes) -> Union[Tuple[IProxyInfo, bytes], Tuple[None, None]]:
        """
        Consume a chunk of data and attempt to parse it.

        @param data: A bytestring.
        @type data: bytes

        @return: A two-tuple containing, in order, an L{IProxyInfo} and any
            bytes fed to the parser that followed the end of the header.  Both
            of these values are None until a complete header is parsed.

        @raises InvalidProxyHeader: If the bytes fed to the parser create an
            invalid PROXY header.
        """

    def parse(line: bytes) -> IProxyInfo:
        """
        Parse a bytestring as a full PROXY protocol header line.

        @param line: A bytestring that represents a valid HAProxy PROXY
            protocol header line.
        @type line: bytes

        @return: An L{IProxyInfo} containing the parsed data.

        @raises InvalidProxyHeader: If the bytestring does not represent a
            valid PROXY header.
        """