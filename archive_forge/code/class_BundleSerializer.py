import base64
import re
from io import BytesIO
from .... import errors, registry
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....timestamp import format_highres_date, unpack_highres_date
class BundleSerializer:
    """The base class for Serializers.

    Common functionality should be included here.
    """

    def __init__(self, version):
        self.version = version

    def read(self, f):
        """Read the rest of the bundles from the supplied file.

        :param f: The file to read from
        :return: A list of bundle trees
        """
        raise NotImplementedError

    def write_bundle(self, repository, target, base, fileobj):
        """Write the bundle to the supplied file.

        :param repository: The repository to retrieve revision data from
        :param target: The revision to provide data for
        :param base: The most recent of ancestor of the revision that does not
            need to be included in the bundle
        :param fileobj: The file to output to
        :return: List of revision ids written
        """
        raise NotImplementedError