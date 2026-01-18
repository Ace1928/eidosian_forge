import os
from paramiko import util
from paramiko.common import byte_mask
from paramiko.ssh_exception import SSHException
def _parse_modulus(self, line):
    timestamp, mod_type, tests, tries, size, generator, modulus = line.split()
    mod_type = int(mod_type)
    tests = int(tests)
    tries = int(tries)
    size = int(size)
    generator = int(generator)
    modulus = int(modulus, 16)
    if mod_type < 2 or tests < 4 or (tests & 4 and tests < 8 and (tries < 100)):
        self.discarded.append((modulus, 'does not meet basic requirements'))
        return
    if generator == 0:
        generator = 2
    bl = util.bit_length(modulus)
    if bl != size and bl != size + 1:
        self.discarded.append((modulus, 'incorrectly reported bit length {}'.format(size)))
        return
    if bl not in self.pack:
        self.pack[bl] = []
    self.pack[bl].append((generator, modulus))