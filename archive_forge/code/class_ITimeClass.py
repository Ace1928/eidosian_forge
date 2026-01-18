from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface import classImplements
class ITimeClass(Interface):
    """This is the time class interface.

    This is symbolic; this module does **not** make
    `datetime.time` provide this interface.

    """
    min = Attribute('The earliest representable time')
    max = Attribute('The latest representable time')
    resolution = Attribute('The smallest possible difference between non-equal time objects')