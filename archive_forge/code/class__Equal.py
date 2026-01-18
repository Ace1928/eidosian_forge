from io import BytesIO
from xml.dom import minidom as dom
from twisted.internet.protocol import FileWrapper
class _Equal:
    """
    A class the instances of which are equal to anything and everything.
    """

    def __eq__(self, other: object) -> bool:
        return True