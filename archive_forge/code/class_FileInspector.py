import struct
from oslo_log import log as logging
class FileInspector(object):
    """A stream-based disk image inspector.

    This base class works on raw images and is subclassed for more
    complex types. It is to be presented with the file to be examined
    one chunk at a time, during read processing and will only store
    as much data as necessary to determine required attributes of
    the file.
    """

    def __init__(self, tracing=False):
        self._total_count = 0
        if tracing:
            self._log = logging.getLogger(str(self))
        else:
            self._log = TraceDisabled()
        self._capture_regions = {}

    def _capture(self, chunk, only=None):
        for name, region in self._capture_regions.items():
            if only and name not in only:
                continue
            if not region.complete:
                region.capture(chunk, self._total_count)

    def eat_chunk(self, chunk):
        """Call this to present chunks of the file to the inspector."""
        pre_regions = set(self._capture_regions.keys())
        self._total_count += len(chunk)
        self._capture(chunk)
        self.post_process()
        new_regions = set(self._capture_regions.keys()) - pre_regions
        if new_regions:
            self._capture(chunk, only=new_regions)

    def post_process(self):
        """Post-read hook to process what has been read so far.

        This will be called after each chunk is read and potentially captured
        by the defined regions. If any regions are defined by this call,
        those regions will be presented with the current chunk in case it
        is within one of the new regions.
        """
        pass

    def region(self, name):
        """Get a CaptureRegion by name."""
        return self._capture_regions[name]

    def new_region(self, name, region):
        """Add a new CaptureRegion by name."""
        if self.has_region(name):
            raise ImageFormatError('Inspector re-added region %s' % name)
        self._capture_regions[name] = region

    def has_region(self, name):
        """Returns True if named region has been defined."""
        return name in self._capture_regions

    @property
    def format_match(self):
        """Returns True if the file appears to be the expected format."""
        return True

    @property
    def virtual_size(self):
        """Returns the virtual size of the disk image, or zero if unknown."""
        return self._total_count

    @property
    def actual_size(self):
        """Returns the total size of the file, usually smaller than
        virtual_size.
        """
        return self._total_count

    def __str__(self):
        """The string name of this file format."""
        return 'raw'

    @property
    def context_info(self):
        """Return info on amount of data held in memory for auditing.

        This is a dict of region:sizeinbytes items that the inspector
        uses to examine the file.
        """
        return {name: len(region.data) for name, region in self._capture_regions.items()}