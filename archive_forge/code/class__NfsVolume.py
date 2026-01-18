import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
@_registered_volume_type
class _NfsVolume(_VolumeType):
    """Volume Type representing an NFS volume."""

    @classmethod
    def name(cls):
        return 'nfs'

    @classmethod
    def help(cls):
        return 'Represents a volume backed by an NFS server.'

    @classmethod
    def required_fields(cls):
        return {'location': 'The location of the NFS Server, in the form SERVER:/PATH'}

    @classmethod
    def optional_fields(cls):
        return {'readonly': 'A boolean. If true, this volume will be read-only from all mounts.'}

    @classmethod
    def fill_volume(cls, volume, new_vol, messages):
        readonly = _is_readonly(volume)
        location = volume['location']
        if ':/' not in location:
            raise serverless_exceptions.ConfigurationError("Volume {}: field 'location' must be of the form IP_ADDRESS:/DIRECTORY".format(volume['name']))
        server, path = location.split(':/', 1)
        path = '/' + path
        src = messages.NFSVolumeSource(server=server, path=path, readOnly=readonly)
        new_vol.nfs = src