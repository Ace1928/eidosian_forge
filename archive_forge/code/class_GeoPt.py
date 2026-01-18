from the atom and gd namespaces. For more information, see:
from __future__ import absolute_import
import base64
import calendar
import datetime
import os
import re
import time
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import namespace_manager
from googlecloudsdk.third_party.appengine.api import users
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
from googlecloudsdk.third_party.appengine.datastore import sortable_pb_encoder
from googlecloudsdk.third_party.appengine._internal import six_subset
class GeoPt(object):
    """A geographical point, specified by floating-point latitude and longitude
  coordinates. Often used to integrate with mapping sites like Google Maps.
  May also be used as ICBM coordinates.

  This is the georss:point element. In XML output, the coordinates are
  provided as the lat and lon attributes. See: http://georss.org/

  Serializes to '<lat>,<lon>'. Raises BadValueError if it's passed an invalid
  serialized string, or if lat and lon are not valid floating points in the
  ranges [-90, 90] and [-180, 180], respectively.
  """
    lat = None
    lon = None

    def __init__(self, lat, lon=None):
        if lon is None:
            try:
                split = lat.split(',')
                lat, lon = split
            except (AttributeError, ValueError):
                raise datastore_errors.BadValueError('Expected a "lat,long" formatted string; received %s (a %s).' % (lat, typename(lat)))
        try:
            lat = float(lat)
            lon = float(lon)
            if abs(lat) > 90:
                raise datastore_errors.BadValueError('Latitude must be between -90 and 90; received %f' % lat)
            if abs(lon) > 180:
                raise datastore_errors.BadValueError('Longitude must be between -180 and 180; received %f' % lon)
        except (TypeError, ValueError):
            raise datastore_errors.BadValueError('Expected floats for lat and long; received %s (a %s) and %s (a %s).' % (lat, typename(lat), lon, typename(lon)))
        self.lat = lat
        self.lon = lon

    def __cmp__(self, other):
        if not isinstance(other, GeoPt):
            try:
                other = GeoPt(other)
            except datastore_errors.BadValueError:
                return NotImplemented
        lat_cmp = cmp(self.lat, other.lat)
        if lat_cmp != 0:
            return lat_cmp
        else:
            return cmp(self.lon, other.lon)

    def __hash__(self):
        """Returns an integer hash of this point.

    Implements Python's hash protocol so that GeoPts may be used in sets and
    as dictionary keys.

    Returns:
      int
    """
        return hash((self.lat, self.lon))

    def __repr__(self):
        """Returns an eval()able string representation of this GeoPt.

    The returned string is of the form 'datastore_types.GeoPt([lat], [lon])'.

    Returns:
      string
    """
        return 'datastore_types.GeoPt(%r, %r)' % (self.lat, self.lon)

    def __unicode__(self):
        return u'%s,%s' % (six_subset.text_type(self.lat), six_subset.text_type(self.lon))
    __str__ = __unicode__

    def ToXml(self):
        return u'<georss:point>%s %s</georss:point>' % (six_subset.text_type(self.lat), six_subset.text_type(self.lon))