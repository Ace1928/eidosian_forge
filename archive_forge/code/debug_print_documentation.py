import sys
from cupy.cuda import memory_hook
Memory hook that prints debug information.

    This memory hook outputs the debug information of input arguments of
    ``malloc`` and ``free`` methods involved in the hooked functions
    at postprocessing time (that is, just after each method is called).

    Example:
        The basic usage is to use it with ``with`` statement.

        Code example::

            >>> import cupy
            >>> from cupy.cuda import memory_hooks
            >>>
            >>> cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
            >>> with memory_hooks.DebugPrintHook():
            ...     x = cupy.array([1, 2, 3])
            ...     del x  # doctest:+SKIP

        Output example::

            {"hook":"alloc","device_id":0,"mem_size":512,"mem_ptr":150496608256}
            {"hook":"malloc","device_id":0,"size":24,"mem_size":512,"mem_ptr":150496608256,"pmem_id":"0x7f39200c5278"}
            {"hook":"free","device_id":0,"mem_size":512,"mem_ptr":150496608256,"pmem_id":"0x7f39200c5278"}

        where the output format is JSONL (JSON Lines) and
        ``hook`` is the name of hook point, and
        ``device_id`` is the CUDA Device ID, and
        ``size`` is the requested memory size to allocate, and
        ``mem_size`` is the rounded memory size to be allocated, and
        ``mem_ptr`` is the memory pointer, and
        ``pmem_id`` is the pooled memory object ID.

    Attributes:
        file: Output file_like object that redirect to.
        flush: If ``True``, this hook forcibly flushes the text stream
            at the end of print. The default is ``True``.

    