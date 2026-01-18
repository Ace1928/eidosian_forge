import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
class GPGGAFixQualities(Values):
    """
    The possible fix quality indications for GPGGA sentences.

    @cvar INVALID_FIX: The fix is invalid.
    @cvar GPS_FIX: There is a fix, acquired using GPS.
    @cvar DGPS_FIX: There is a fix, acquired using differential GPS (DGPS).
    @cvar PPS_FIX: There is a fix, acquired using the precise positioning
        service (PPS).
    @cvar RTK_FIX: There is a fix, acquired using fixed real-time
        kinematics. This means that there was a sufficient number of shared
        satellites with the base station, usually yielding a resolution in
        the centimeter range. This was added in NMEA 0183 version 3.0. This
        is also called Carrier-Phase Enhancement or CPGPS, particularly when
        used in combination with GPS.
    @cvar FLOAT_RTK_FIX: There is a fix, acquired using floating real-time
        kinematics. The same comments apply as for a fixed real-time
        kinematics fix, except that there were insufficient shared satellites
        to acquire it, so instead you got a slightly less good floating fix.
        Typical resolution in the decimeter range.
    @cvar DEAD_RECKONING: There is currently no more fix, but this data was
        computed using a previous fix and some information about motion
        (either from that fix or from other sources) using simple dead
        reckoning. Not particularly reliable, but better-than-nonsense data.
    @cvar MANUAL: There is no real fix from this device, but the location has
        been manually entered, presumably with data obtained from some other
        positioning method.
    @cvar SIMULATED: There is no real fix, but instead it is being simulated.
    """
    INVALID_FIX = '0'
    GPS_FIX = '1'
    DGPS_FIX = '2'
    PPS_FIX = '3'
    RTK_FIX = '4'
    FLOAT_RTK_FIX = '5'
    DEAD_RECKONING = '6'
    MANUAL = '7'
    SIMULATED = '8'