from __future__ import with_statement
from winappdbg.textio import HexDump
from winappdbg import win32
import ctypes
import warnings
class Disassembler(object):
    """
    Generic disassembler. Uses a set of adapters to decide which library to
    load for which supported platform.

    @type engines: tuple( L{Engine} )
    @cvar engines: Set of supported engines. If you implement your own adapter
        you can add its class here to make it available to L{Disassembler}.
        Supported disassemblers are:
    """
    engines = (DistormEngine, BeaEngine, CapstoneEngine, LibdisassembleEngine, PyDasmEngine)
    __doc__ += '\n'
    for e in engines:
        __doc__ += '         - %s - %s (U{%s})\n' % (e.name, e.desc, e.url)
    del e
    __decoder = {}

    def __new__(cls, arch=None, engine=None):
        """
        Factory class. You can't really instance a L{Disassembler} object,
        instead one of the adapter L{Engine} subclasses is returned.

        @type  arch: str
        @param arch: (Optional) Name of the processor architecture.
            If not provided the current processor architecture is assumed.
            For more details see L{win32.version._get_arch}.

        @type  engine: str
        @param engine: (Optional) Name of the disassembler engine.
            If not provided a compatible one is loaded automatically.
            See: L{Engine.name}

        @raise NotImplementedError: No compatible disassembler was found that
            could decode machine code for the requested architecture. This may
            be due to missing dependencies.

        @raise ValueError: An unknown engine name was supplied.
        """
        if not arch:
            arch = win32.arch
        if not engine:
            found = False
            for clazz in cls.engines:
                try:
                    if arch in clazz.supported:
                        selected = (clazz.name, arch)
                        try:
                            decoder = cls.__decoder[selected]
                        except KeyError:
                            decoder = clazz(arch)
                            cls.__decoder[selected] = decoder
                        return decoder
                except NotImplementedError:
                    pass
            msg = 'No disassembler engine available for %s code.' % arch
            raise NotImplementedError(msg)
        selected = (engine, arch)
        try:
            decoder = cls.__decoder[selected]
        except KeyError:
            found = False
            engineLower = engine.lower()
            for clazz in cls.engines:
                if clazz.name.lower() == engineLower:
                    found = True
                    break
            if not found:
                msg = 'Unsupported disassembler engine: %s' % engine
                raise ValueError(msg)
            if arch not in clazz.supported:
                msg = 'The %s engine cannot decode %s code.' % selected
                raise NotImplementedError(msg)
            decoder = clazz(arch)
            cls.__decoder[selected] = decoder
        return decoder