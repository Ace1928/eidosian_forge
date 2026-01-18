import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
class IPIProtocol:
    """Communication using IPI protocol."""

    def __init__(self, socket, txt=None):
        self.socket = socket
        if txt is None:

            def log(*args):
                pass
        else:

            def log(*args):
                print('Driver:', *args, file=txt)
                txt.flush()
        self.log = log

    def sendmsg(self, msg):
        self.log('  sendmsg', repr(msg))
        msg = msg.encode('ascii').ljust(12)
        self.socket.sendall(msg)

    def _recvall(self, nbytes):
        """Repeatedly read chunks until we have nbytes.

        Normally we get all bytes in one read, but that is not guaranteed."""
        remaining = nbytes
        chunks = []
        while remaining > 0:
            chunk = self.socket.recv(remaining)
            if len(chunk) == 0:
                raise SocketClosed()
            chunks.append(chunk)
            remaining -= len(chunk)
        msg = b''.join(chunks)
        assert len(msg) == nbytes and remaining == 0
        return msg

    def recvmsg(self):
        msg = self._recvall(12)
        if not msg:
            raise SocketClosed()
        assert len(msg) == 12, msg
        msg = msg.rstrip().decode('ascii')
        self.log('  recvmsg', repr(msg))
        return msg

    def send(self, a, dtype):
        buf = np.asarray(a, dtype).tobytes()
        self.log('  send {} bytes of {}'.format(len(buf), dtype))
        self.socket.sendall(buf)

    def recv(self, shape, dtype):
        a = np.empty(shape, dtype)
        nbytes = np.dtype(dtype).itemsize * np.prod(shape)
        buf = self._recvall(nbytes)
        assert len(buf) == nbytes, (len(buf), nbytes)
        self.log('  recv {} bytes of {}'.format(len(buf), dtype))
        a.flat[:] = np.frombuffer(buf, dtype=dtype)
        assert np.isfinite(a).all()
        return a

    def sendposdata(self, cell, icell, positions):
        assert cell.size == 9
        assert icell.size == 9
        assert positions.size % 3 == 0
        self.log(' sendposdata')
        self.sendmsg('POSDATA')
        self.send(cell.T / units.Bohr, np.float64)
        self.send(icell.T * units.Bohr, np.float64)
        self.send(len(positions), np.int32)
        self.send(positions / units.Bohr, np.float64)

    def recvposdata(self):
        cell = self.recv((3, 3), np.float64).T.copy()
        icell = self.recv((3, 3), np.float64).T.copy()
        natoms = self.recv(1, np.int32)
        natoms = int(natoms)
        positions = self.recv((natoms, 3), np.float64)
        return (cell * units.Bohr, icell / units.Bohr, positions * units.Bohr)

    def sendrecv_force(self):
        self.log(' sendrecv_force')
        self.sendmsg('GETFORCE')
        msg = self.recvmsg()
        assert msg == 'FORCEREADY', msg
        e = self.recv(1, np.float64)[0]
        natoms = self.recv(1, np.int32)
        assert natoms >= 0
        forces = self.recv((int(natoms), 3), np.float64)
        virial = self.recv((3, 3), np.float64).T.copy()
        nmorebytes = self.recv(1, np.int32)
        nmorebytes = int(nmorebytes)
        if nmorebytes > 0:
            morebytes = self.recv(nmorebytes, np.byte)
        else:
            morebytes = b''
        return (e * units.Ha, units.Ha / units.Bohr * forces, units.Ha * virial, morebytes)

    def sendforce(self, energy, forces, virial, morebytes=np.zeros(1, dtype=np.byte)):
        assert np.array([energy]).size == 1
        assert forces.shape[1] == 3
        assert virial.shape == (3, 3)
        self.log(' sendforce')
        self.sendmsg('FORCEREADY')
        self.send(np.array([energy / units.Ha]), np.float64)
        natoms = len(forces)
        self.send(np.array([natoms]), np.int32)
        self.send(units.Bohr / units.Ha * forces, np.float64)
        self.send(1.0 / units.Ha * virial.T, np.float64)
        self.send(np.array([len(morebytes)]), np.int32)
        self.send(morebytes, np.byte)

    def status(self):
        self.log(' status')
        self.sendmsg('STATUS')
        msg = self.recvmsg()
        return msg

    def end(self):
        self.log(' end')
        self.sendmsg('EXIT')

    def recvinit(self):
        self.log(' recvinit')
        bead_index = self.recv(1, np.int32)
        nbytes = self.recv(1, np.int32)
        initbytes = self.recv(nbytes, np.byte)
        return (bead_index, initbytes)

    def sendinit(self):
        self.log(' sendinit')
        self.sendmsg('INIT')
        self.send(0, np.int32)
        self.send(1, np.int32)
        self.send(np.zeros(1), np.byte)

    def calculate(self, positions, cell):
        self.log('calculate')
        msg = self.status()
        if msg == 'NEEDINIT':
            self.sendinit()
            msg = self.status()
        assert msg == 'READY', msg
        icell = np.linalg.pinv(cell).transpose()
        self.sendposdata(cell, icell, positions)
        msg = self.status()
        assert msg == 'HAVEDATA', msg
        e, forces, virial, morebytes = self.sendrecv_force()
        r = dict(energy=e, forces=forces, virial=virial)
        if morebytes:
            r['morebytes'] = morebytes
        return r