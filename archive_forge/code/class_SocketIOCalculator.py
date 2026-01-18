import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
class SocketIOCalculator(Calculator, IOContext):
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']
    supported_changes = {'positions', 'cell'}

    def __init__(self, calc=None, port=None, unixsocket=None, timeout=None, log=None, *, launch_client=None):
        """Initialize socket I/O calculator.

        This calculator launches a server which passes atomic
        coordinates and unit cells to an external code via a socket,
        and receives energy, forces, and stress in return.

        ASE integrates this with the Quantum Espresso, FHI-aims and
        Siesta calculators.  This works with any external code that
        supports running as a client over the i-PI protocol.

        Parameters:

        calc: calculator or None

            If calc is not None, a client process will be launched
            using calc.command, and the input file will be generated
            using ``calc.write_input()``.  Otherwise only the server will
            run, and it is up to the user to launch a compliant client
            process.

        port: integer

            port number for socket.  Should normally be between 1025
            and 65535.  Typical ports for are 31415 (default) or 3141.

        unixsocket: str or None

            if not None, ignore host and port, creating instead a
            unix socket using this name prefixed with ``/tmp/ipi_``.
            The socket is deleted when the calculator is closed.

        timeout: float >= 0 or None

            timeout for connection, by default infinite.  See
            documentation of Python sockets.  For longer jobs it is
            recommended to set a timeout in case of undetected
            client-side failure.

        log: file object or None (default)

            logfile for communication over socket.  For debugging or
            the curious.

        In order to correctly close the sockets, it is
        recommended to use this class within a with-block:

        >>> with SocketIOCalculator(...) as calc:
        ...    atoms.calc = calc
        ...    atoms.get_forces()
        ...    atoms.rattle()
        ...    atoms.get_forces()

        It is also possible to call calc.close() after
        use.  This is best done in a finally-block."""
        Calculator.__init__(self)
        if calc is not None:
            if launch_client is not None:
                raise ValueError('Cannot pass both calc and launch_client')
            launch_client = FileIOSocketClientLauncher(calc)
        self.launch_client = launch_client
        self.timeout = timeout
        self.server = None
        self.log = self.openfile(log)
        self._port = port
        self._unixsocket = unixsocket
        if self.launch_client is None:
            self.server = self.launch_server()

    def todict(self):
        d = {'type': 'calculator', 'name': 'socket-driver'}
        return d

    def launch_server(self):
        return self.closelater(SocketServer(port=self._port, unixsocket=self._unixsocket, timeout=self.timeout, log=self.log))

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        bad = [change for change in system_changes if change not in self.supported_changes]
        if self.atoms is not None and any(bad):
            raise PropertyNotImplementedError('Cannot change {} through IPI protocol.  Please create new socket calculator.'.format(bad if len(bad) > 1 else bad[0]))
        self.atoms = atoms.copy()
        if self.server is None:
            self.server = self.launch_server()
            proc = self.launch_client(atoms, properties, port=self._port, unixsocket=self._unixsocket)
            self.server.proc = proc
        results = self.server.calculate(atoms)
        results['free_energy'] = results['energy']
        virial = results.pop('virial')
        if self.atoms.cell.rank == 3 and any(self.atoms.pbc):
            vol = atoms.get_volume()
            results['stress'] = -full_3x3_to_voigt_6_stress(virial) / vol
        self.results.update(results)

    def close(self):
        self.server = None
        super().close()