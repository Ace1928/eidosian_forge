from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
class CoordinateTests(TestCase):

    def test_float(self) -> None:
        """
        Coordinates can be converted to floats.
        """
        coordinate = base.Coordinate(10.0)
        self.assertEqual(float(coordinate), 10.0)

    def test_repr(self) -> None:
        """
        Coordinates that aren't explicitly latitudes or longitudes have an
        appropriate repr.
        """
        coordinate = base.Coordinate(10.0)
        expectedRepr = f'<Angle of unknown type ({10.0} degrees)>'
        self.assertEqual(repr(coordinate), expectedRepr)

    def test_positiveLatitude(self) -> None:
        """
        Positive latitudes have a repr that specifies their type and value.
        """
        coordinate = base.Coordinate(10.0, Angles.LATITUDE)
        expectedRepr = f'<Latitude ({10.0} degrees)>'
        self.assertEqual(repr(coordinate), expectedRepr)

    def test_negativeLatitude(self) -> None:
        """
        Negative latitudes have a repr that specifies their type and value.
        """
        coordinate = base.Coordinate(-50.0, Angles.LATITUDE)
        expectedRepr = f'<Latitude ({-50.0} degrees)>'
        self.assertEqual(repr(coordinate), expectedRepr)

    def test_positiveLongitude(self) -> None:
        """
        Positive longitudes have a repr that specifies their type and value.
        """
        longitude = base.Coordinate(50.0, Angles.LONGITUDE)
        expectedRepr = f'<Longitude ({50.0} degrees)>'
        self.assertEqual(repr(longitude), expectedRepr)

    def test_negativeLongitude(self) -> None:
        """
        Negative longitudes have a repr that specifies their type and value.
        """
        longitude = base.Coordinate(-50.0, Angles.LONGITUDE)
        expectedRepr = f'<Longitude ({-50.0} degrees)>'
        self.assertEqual(repr(longitude), expectedRepr)

    def test_bogusCoordinateType(self) -> None:
        """
        Creating coordinates with bogus types rasies C{ValueError}.
        """
        self.assertRaises(ValueError, base.Coordinate, 150.0, 'BOGUS')

    def test_angleTypeNotCoordinate(self) -> None:
        """
        Creating coordinates with angle types that aren't coordinates raises
        C{ValueError}.
        """
        self.assertRaises(ValueError, base.Coordinate, 150.0, Angles.HEADING)

    def test_equality(self) -> None:
        """
        Coordinates with the same value and type are equal.
        """

        def makeCoordinate() -> base.Coordinate:
            return base.Coordinate(1.0, Angles.LONGITUDE)
        self.assertEqual(makeCoordinate(), makeCoordinate())

    def test_differentAnglesInequality(self) -> None:
        """
        Coordinates with different values aren't equal.
        """
        c1 = base.Coordinate(1.0)
        c2 = base.Coordinate(-1.0)
        self.assertNotEqual(c1, c2)

    def test_differentTypesInequality(self) -> None:
        """
        Coordinates with the same values but different types aren't equal.
        """
        c1 = base.Coordinate(1.0, Angles.LATITUDE)
        c2 = base.Coordinate(1.0, Angles.LONGITUDE)
        self.assertNotEqual(c1, c2)

    def test_sign(self) -> None:
        """
        Setting the sign on a coordinate sets the sign of the value of the
        coordinate.
        """
        c = base.Coordinate(50.0, Angles.LATITUDE)
        c.setSign(1)
        self.assertEqual(c.inDecimalDegrees, 50.0)
        c.setSign(-1)
        self.assertEqual(c.inDecimalDegrees, -50.0)

    def test_badVariationSign(self) -> None:
        """
        Setting a bogus sign value (not -1 or 1) on a coordinate raises
        C{ValueError} and doesn't affect the coordinate.
        """
        value = 50.0
        c = base.Coordinate(value, Angles.LATITUDE)
        self.assertRaises(ValueError, c.setSign, -50)
        self.assertEqual(c.inDecimalDegrees, 50.0)
        self.assertRaises(ValueError, c.setSign, 0)
        self.assertEqual(c.inDecimalDegrees, 50.0)
        self.assertRaises(ValueError, c.setSign, 50)
        self.assertEqual(c.inDecimalDegrees, 50.0)

    def test_northernHemisphere(self) -> None:
        """
        Positive latitudes are in the northern hemisphere.
        """
        coordinate = base.Coordinate(1.0, Angles.LATITUDE)
        self.assertEqual(coordinate.hemisphere, Directions.NORTH)

    def test_easternHemisphere(self) -> None:
        """
        Positive longitudes are in the eastern hemisphere.
        """
        coordinate = base.Coordinate(1.0, Angles.LONGITUDE)
        self.assertEqual(coordinate.hemisphere, Directions.EAST)

    def test_southernHemisphere(self) -> None:
        """
        Negative latitudes are in the southern hemisphere.
        """
        coordinate = base.Coordinate(-1.0, Angles.LATITUDE)
        self.assertEqual(coordinate.hemisphere, Directions.SOUTH)

    def test_westernHemisphere(self) -> None:
        """
        Negative longitudes are in the western hemisphere.
        """
        coordinate = base.Coordinate(-1.0, Angles.LONGITUDE)
        self.assertEqual(coordinate.hemisphere, Directions.WEST)

    def test_badHemisphere(self) -> None:
        """
        Accessing the hemisphere for a coordinate that can't compute it
        raises C{ValueError}.
        """
        coordinate = base.Coordinate(1.0, None)
        self.assertRaises(ValueError, lambda: coordinate.hemisphere)

    def test_latitudeTooLarge(self) -> None:
        """
        Creating a latitude with a value greater than or equal to 90 degrees
        raises C{ValueError}.
        """
        self.assertRaises(ValueError, _makeLatitude, 150.0)
        self.assertRaises(ValueError, _makeLatitude, 90.0)

    def test_latitudeTooSmall(self) -> None:
        """
        Creating a latitude with a value less than or equal to -90 degrees
        raises C{ValueError}.
        """
        self.assertRaises(ValueError, _makeLatitude, -150.0)
        self.assertRaises(ValueError, _makeLatitude, -90.0)

    def test_longitudeTooLarge(self) -> None:
        """
        Creating a longitude with a value greater than or equal to 180 degrees
        raises C{ValueError}.
        """
        self.assertRaises(ValueError, _makeLongitude, 250.0)
        self.assertRaises(ValueError, _makeLongitude, 180.0)

    def test_longitudeTooSmall(self) -> None:
        """
        Creating a longitude with a value less than or equal to -180 degrees
        raises C{ValueError}.
        """
        self.assertRaises(ValueError, _makeLongitude, -250.0)
        self.assertRaises(ValueError, _makeLongitude, -180.0)

    def test_inDegreesMinutesSeconds(self) -> None:
        """
        Coordinate values can be accessed in degrees, minutes, seconds.
        """
        c = base.Coordinate(50.5, Angles.LATITUDE)
        self.assertEqual(c.inDegreesMinutesSeconds, (50, 30, 0))
        c = base.Coordinate(50.213, Angles.LATITUDE)
        self.assertEqual(c.inDegreesMinutesSeconds, (50, 12, 46))

    def test_unknownAngleInDegreesMinutesSeconds(self) -> None:
        """
        If the vaue of a coordinate is L{None}, its values in degrees,
        minutes, seconds is also L{None}.
        """
        c = base.Coordinate(None, None)
        self.assertIsNone(c.inDegreesMinutesSeconds)