import struct
from oslo_log import log as logging
def _find_meta_entry(self, desired_guid):
    meta_buffer = self.region('metadata').data
    if len(meta_buffer) < 32:
        return None
    sig, reserved, count = struct.unpack('<8sHH', meta_buffer[:12])
    if sig != b'metadata':
        raise ImageFormatError('Invalid signature for metadata region: %r' % sig)
    entries_size = 32 + count * 32
    if len(meta_buffer) < entries_size:
        return None
    if count >= 2048:
        raise ImageFormatError('Metadata item count is %i (limit 2047)' % count)
    for i in range(0, count):
        entry_offset = 32 + i * 32
        guid = self._guid(meta_buffer[entry_offset:entry_offset + 16])
        if guid == desired_guid:
            item_offset, item_length, _reserved = struct.unpack('<III', meta_buffer[entry_offset + 16:entry_offset + 28])
            item_length = min(item_length, self.VHDX_METADATA_TABLE_MAX_SIZE)
            self.region('metadata').length = len(meta_buffer)
            self._log.debug('Found entry at offset %x', item_offset)
            return CaptureRegion(self.region('metadata').offset + item_offset, item_length)
    self._log.warning('Did not find guid %s', desired_guid)
    return None