import os.path
from ._paml import Paml
from . import _parse_yn00
class Yn00(Paml):
    """An interface to yn00, part of the PAML package."""

    def __init__(self, alignment=None, working_dir=None, out_file=None):
        """Initialize the Yn00 instance.

        The user may optionally pass in strings specifying the locations
        of the input alignment, the working directory and
        the final output file.
        """
        Paml.__init__(self, alignment, working_dir, out_file)
        self.ctl_file = 'yn00.ctl'
        self._options = {'verbose': None, 'icode': None, 'weighting': None, 'commonf3x4': None, 'ndata': None}

    def write_ctl_file(self):
        """Dynamically build a yn00 control file from the options.

        The control file is written to the location specified by the
        ctl_file property of the yn00 class.
        """
        self._set_rel_paths()
        with open(self.ctl_file, 'w') as ctl_handle:
            ctl_handle.write(f'seqfile = {self._rel_alignment}\n')
            ctl_handle.write(f'outfile = {self._rel_out_file}\n')
            for option in self._options.items():
                if option[1] is None:
                    continue
                ctl_handle.write(f'{option[0]} = {option[1]}\n')

    def read_ctl_file(self, ctl_file):
        """Parse a control file and load the options into the yn00 instance."""
        temp_options = {}
        if not os.path.isfile(ctl_file):
            raise FileNotFoundError(f'File not found: {ctl_file!r}')
        else:
            with open(ctl_file) as ctl_handle:
                for line in ctl_handle:
                    line = line.strip()
                    uncommented = line.split('*', 1)[0]
                    if uncommented != '':
                        if '=' not in uncommented:
                            raise AttributeError(f'Malformed line in control file:\n{line!r}')
                        option, value = uncommented.split('=')
                        option = option.strip()
                        value = value.strip()
                        if option == 'seqfile':
                            self.alignment = value
                        elif option == 'outfile':
                            self.out_file = value
                        elif option not in self._options:
                            raise KeyError(f'Invalid option: {option}')
                        else:
                            if '.' in value or 'e-' in value:
                                try:
                                    converted_value = float(value)
                                except ValueError:
                                    converted_value = value
                            else:
                                try:
                                    converted_value = int(value)
                                except ValueError:
                                    converted_value = value
                            temp_options[option] = converted_value
        for option in self._options:
            if option in temp_options:
                self._options[option] = temp_options[option]
            else:
                self._options[option] = None

    def run(self, ctl_file=None, verbose=False, command='yn00', parse=True):
        """Run yn00 using the current configuration.

        If parse is True then read and return the result, otherwise
        return None.
        """
        Paml.run(self, ctl_file, verbose, command)
        if parse:
            return read(self.out_file)
        return None