from zunclient.common import base
class AvailabilityZone(base.Resource):

    def __repr__(self):
        return '<AvailabilityZone %s>' % self._info