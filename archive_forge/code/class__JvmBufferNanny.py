import pyarrow as pa
class _JvmBufferNanny:
    """
    An object that keeps a org.apache.arrow.memory.ArrowBuf's underlying
    memory alive.
    """
    ref_manager = None

    def __init__(self, jvm_buf):
        ref_manager = jvm_buf.getReferenceManager()
        ref_manager.retain()
        self.ref_manager = ref_manager

    def __del__(self):
        if self.ref_manager is not None:
            self.ref_manager.release()