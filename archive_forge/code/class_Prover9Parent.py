import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
class Prover9Parent:
    """
    A common class extended by both ``Prover9`` and ``Mace <mace.Mace>``.
    It contains the functionality required to convert NLTK-style
    expressions into Prover9-style expressions.
    """
    _binary_location = None

    def config_prover9(self, binary_location, verbose=False):
        if binary_location is None:
            self._binary_location = None
            self._prover9_bin = None
        else:
            name = 'prover9'
            self._prover9_bin = nltk.internals.find_binary(name, path_to_bin=binary_location, env_vars=['PROVER9'], url='https://www.cs.unm.edu/~mccune/prover9/', binary_names=[name, name + '.exe'], verbose=verbose)
            self._binary_location = self._prover9_bin.rsplit(os.path.sep, 1)

    def prover9_input(self, goal, assumptions):
        """
        :return: The input string that should be provided to the
            prover9 binary.  This string is formed based on the goal,
            assumptions, and timeout value of this object.
        """
        s = ''
        if assumptions:
            s += 'formulas(assumptions).\n'
            for p9_assumption in convert_to_prover9(assumptions):
                s += '    %s.\n' % p9_assumption
            s += 'end_of_list.\n\n'
        if goal:
            s += 'formulas(goals).\n'
            s += '    %s.\n' % convert_to_prover9(goal)
            s += 'end_of_list.\n\n'
        return s

    def binary_locations(self):
        """
        A list of directories that should be searched for the prover9
        executables.  This list is used by ``config_prover9`` when searching
        for the prover9 executables.
        """
        return ['/usr/local/bin/prover9', '/usr/local/bin/prover9/bin', '/usr/local/bin', '/usr/bin', '/usr/local/prover9', '/usr/local/share/prover9']

    def _find_binary(self, name, verbose=False):
        binary_locations = self.binary_locations()
        if self._binary_location is not None:
            binary_locations += [self._binary_location]
        return nltk.internals.find_binary(name, searchpath=binary_locations, env_vars=['PROVER9'], url='https://www.cs.unm.edu/~mccune/prover9/', binary_names=[name, name + '.exe'], verbose=verbose)

    def _call(self, input_str, binary, args=[], verbose=False):
        """
        Call the binary with the given input.

        :param input_str: A string whose contents are used as stdin.
        :param binary: The location of the binary to call
        :param args: A list of command-line arguments.
        :return: A tuple (stdout, returncode)
        :see: ``config_prover9``
        """
        if verbose:
            print('Calling:', binary)
            print('Args:', args)
            print('Input:\n', input_str, '\n')
        cmd = [binary] + args
        try:
            input_str = input_str.encode('utf8')
        except AttributeError:
            pass
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
        stdout, stderr = p.communicate(input=input_str)
        if verbose:
            print('Return code:', p.returncode)
            if stdout:
                print('stdout:\n', stdout, '\n')
            if stderr:
                print('stderr:\n', stderr, '\n')
        return (stdout.decode('utf-8'), p.returncode)