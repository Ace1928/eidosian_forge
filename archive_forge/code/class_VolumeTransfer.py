from cinderclient import base
class VolumeTransfer(base.Resource):
    """Transfer a volume from one tenant to another"""

    def __repr__(self):
        return '<VolumeTransfer: %s>' % self.id

    def delete(self):
        """Delete this volume transfer."""
        return self.manager.delete(self)