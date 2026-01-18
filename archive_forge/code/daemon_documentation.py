from socket import AF_UNSPEC as _AF_UNSPEC
from ._daemon import (__version__,
Return a dictionary of socket activated descriptors as {fd: name}

    Example::

      (in primary window)
      $ systemd-socket-activate -l 2000 -l 4000 --fdname=2K:4K python3 -c \
          'from systemd.daemon import listen_fds_with_names; print(listen_fds_with_names())'
      (in another window)
      $ telnet localhost 2000
      (in primary window)
      ...
      Execing python3 (...)
      [3]
    