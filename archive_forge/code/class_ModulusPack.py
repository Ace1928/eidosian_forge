import os
from paramiko import util
from paramiko.common import byte_mask
from paramiko.ssh_exception import SSHException
class ModulusPack:
    """
    convenience object for holding the contents of the /etc/ssh/moduli file,
    on systems that have such a file.
    """

    def __init__(self):
        self.pack = {}
        self.discarded = []

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

    def read_file(self, filename):
        """
        :raises IOError: passed from any file operations that fail.
        """
        self.pack = {}
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line[0] == '#':
                    continue
                try:
                    self._parse_modulus(line)
                except:
                    continue

    def get_modulus(self, min, prefer, max):
        bitsizes = sorted(self.pack.keys())
        if len(bitsizes) == 0:
            raise SSHException('no moduli available')
        good = -1
        for b in bitsizes:
            if b >= prefer and b <= max and (b < good or good == -1):
                good = b
        if good == -1:
            for b in bitsizes:
                if b >= min and b <= max and (b > good):
                    good = b
        if good == -1:
            good = bitsizes[0]
            if min > good:
                good = bitsizes[-1]
        n = _roll_random(len(self.pack[good]))
        return self.pack[good][n]