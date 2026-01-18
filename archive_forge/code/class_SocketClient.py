import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
class SocketClient:

    def __init__(self, host='localhost', port=None, unixsocket=None, timeout=None, log=None, comm=None):
        """Create client and connect to server.

        Parameters:

        host: string
            Hostname of server.  Defaults to localhost
        port: integer or None
            Port to which to connect.  By default 31415.
        unixsocket: string or None
            If specified, use corresponding UNIX socket.
            See documentation of unixsocket for SocketIOCalculator.
        timeout: float or None
            See documentation of timeout for SocketIOCalculator.
        log: file object or None
            Log events to this file
        comm: communicator or None
            MPI communicator object.  Defaults to ase.parallel.world.
            When ASE runs in parallel, only the process with world.rank == 0
            will communicate over the socket.  The received information
            will then be broadcast on the communicator.  The SocketClient
            must be created on all ranks of world, and will see the same
            Atoms objects."""
        if comm is None:
            from ase.parallel import world
            comm = world
        self.comm = comm
        if self.comm.rank == 0:
            if unixsocket is not None:
                sock = socket.socket(socket.AF_UNIX)
                actualsocket = actualunixsocketname(unixsocket)
                sock.connect(actualsocket)
            else:
                if port is None:
                    port = SocketServer.default_port
                sock = socket.socket(socket.AF_INET)
                sock.connect((host, port))
            sock.settimeout(timeout)
            self.host = host
            self.port = port
            self.unixsocket = unixsocket
            self.protocol = IPIProtocol(sock, txt=log)
            self.log = self.protocol.log
            self.closed = False
            self.bead_index = 0
            self.bead_initbytes = b''
            self.state = 'READY'

    def close(self):
        if not self.closed:
            self.log('Close SocketClient')
            self.closed = True
            self.protocol.socket.close()

    def calculate(self, atoms, use_stress):
        self.comm.broadcast(atoms.positions, 0)
        self.comm.broadcast(np.ascontiguousarray(atoms.cell), 0)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        if use_stress:
            stress = atoms.get_stress(voigt=False)
            virial = -atoms.get_volume() * stress
        else:
            virial = np.zeros((3, 3))
        return (energy, forces, virial)

    def irun(self, atoms, use_stress=None):
        if use_stress is None:
            use_stress = any(atoms.pbc)
        my_irun = self.irun_rank0 if self.comm.rank == 0 else self.irun_rankN
        return my_irun(atoms, use_stress)

    def irun_rankN(self, atoms, use_stress=True):
        stop_criterion = np.zeros(1, bool)
        while True:
            self.comm.broadcast(stop_criterion, 0)
            if stop_criterion[0]:
                return
            self.calculate(atoms, use_stress)
            yield

    def irun_rank0(self, atoms, use_stress=True):
        try:
            while True:
                try:
                    msg = self.protocol.recvmsg()
                except SocketClosed:
                    msg = 'EXIT'
                if msg == 'EXIT':
                    self.comm.broadcast(np.ones(1, bool), 0)
                    return
                elif msg == 'STATUS':
                    self.protocol.sendmsg(self.state)
                elif msg == 'POSDATA':
                    assert self.state == 'READY'
                    cell, icell, positions = self.protocol.recvposdata()
                    atoms.cell[:] = cell
                    atoms.positions[:] = positions
                    self.comm.broadcast(np.zeros(1, bool), 0)
                    energy, forces, virial = self.calculate(atoms, use_stress)
                    self.state = 'HAVEDATA'
                    yield
                elif msg == 'GETFORCE':
                    assert self.state == 'HAVEDATA', self.state
                    self.protocol.sendforce(energy, forces, virial)
                    self.state = 'NEEDINIT'
                elif msg == 'INIT':
                    assert self.state == 'NEEDINIT'
                    bead_index, initbytes = self.protocol.recvinit()
                    self.bead_index = bead_index
                    self.bead_initbytes = initbytes
                    self.state = 'READY'
                else:
                    raise KeyError('Bad message', msg)
        finally:
            self.close()

    def run(self, atoms, use_stress=False):
        for _ in self.irun(atoms, use_stress=use_stress):
            pass