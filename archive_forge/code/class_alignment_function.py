import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
class alignment_function:
    """Callable class which impersonates an alignment function.

        The constructor takes the name of the function.  This class
        will decode the name of the function to figure out how to
        interpret the parameters.
        """
    match2args = {'x': ([], ''), 'm': (['match', 'mismatch'], 'match is the score to given to identical characters.\nmismatch is the score given to non-identical ones.'), 'd': (['match_dict'], "match_dict is a dictionary where the keys are tuples\nof pairs of characters and the values are the scores,\ne.g. ('A', 'C') : 2.5."), 'c': (['match_fn'], 'match_fn is a callback function that takes two characters and returns the score between them.')}
    penalty2args = {'x': ([], ''), 's': (['open', 'extend'], 'open and extend are the gap penalties when a gap is\nopened and extended.  They should be negative.'), 'd': (['openA', 'extendA', 'openB', 'extendB'], 'openA and extendA are the gap penalties for sequenceA,\nand openB and extendB for sequenceB.  The penalties\nshould be negative.'), 'c': (['gap_A_fn', 'gap_B_fn'], 'gap_A_fn and gap_B_fn are callback functions that takes\n(1) the index where the gap is opened, and (2) the length\nof the gap.  They should return a gap penalty.')}

    def __init__(self, name):
        """Check to make sure the name of the function is reasonable."""
        if name.startswith('global'):
            if len(name) != 8:
                raise AttributeError('function should be globalXX')
        elif name.startswith('local'):
            if len(name) != 7:
                raise AttributeError('function should be localXX')
        else:
            raise AttributeError(name)
        align_type, match_type, penalty_type = (name[:-2], name[-2], name[-1])
        try:
            match_args, match_doc = self.match2args[match_type]
        except KeyError:
            raise AttributeError(f'unknown match type {match_type!r}')
        try:
            penalty_args, penalty_doc = self.penalty2args[penalty_type]
        except KeyError:
            raise AttributeError(f'unknown penalty type {penalty_type!r}')
        param_names = ['sequenceA', 'sequenceB']
        param_names.extend(match_args)
        param_names.extend(penalty_args)
        self.function_name = name
        self.align_type = align_type
        self.param_names = param_names
        self.__name__ = self.function_name
        doc = f'{self.__name__}({', '.join(self.param_names)}) -> alignments\n'
        doc += '\nThe following parameters can also be used with optional\nkeywords of the same name.\n\n\nsequenceA and sequenceB must be of the same type, either\nstrings, lists or Biopython sequence objects.\n\n'
        if match_doc:
            doc += f'\n{match_doc}\n'
        if penalty_doc:
            doc += f'\n{penalty_doc}\n'
        doc += '\nalignments is a list of named tuples (seqA, seqB, score,\nbegin, end). seqA and seqB are strings showing the alignment\nbetween the sequences.  score is the score of the alignment.\nbegin and end are indexes of seqA and seqB that indicate\nwhere the alignment occurs.\n'
        self.__doc__ = doc

    def decode(self, *args, **keywds):
        """Decode the arguments for the _align function.

            keywds will get passed to it, so translate the arguments
            to this function into forms appropriate for _align.
            """
        keywds = keywds.copy()
        args += (len(self.param_names) - len(args)) * (None,)
        for key in keywds.copy():
            if key in self.param_names:
                _index = self.param_names.index(key)
                args = args[:_index] + (keywds[key],) + args[_index:]
                del keywds[key]
        args = tuple((arg for arg in args if arg is not None))
        if len(args) != len(self.param_names):
            raise TypeError('%s takes exactly %d argument (%d given)' % (self.function_name, len(self.param_names), len(args)))
        i = 0
        while i < len(self.param_names):
            if self.param_names[i] in ['sequenceA', 'sequenceB', 'gap_A_fn', 'gap_B_fn', 'match_fn']:
                keywds[self.param_names[i]] = args[i]
                i += 1
            elif self.param_names[i] == 'match':
                assert self.param_names[i + 1] == 'mismatch'
                match, mismatch = (args[i], args[i + 1])
                keywds['match_fn'] = identity_match(match, mismatch)
                i += 2
            elif self.param_names[i] == 'match_dict':
                keywds['match_fn'] = dictionary_match(args[i])
                i += 1
            elif self.param_names[i] == 'open':
                assert self.param_names[i + 1] == 'extend'
                open, extend = (args[i], args[i + 1])
                pe = keywds.get('penalize_extend_when_opening', 0)
                keywds['gap_A_fn'] = affine_penalty(open, extend, pe)
                keywds['gap_B_fn'] = affine_penalty(open, extend, pe)
                i += 2
            elif self.param_names[i] == 'openA':
                assert self.param_names[i + 3] == 'extendB'
                openA, extendA, openB, extendB = args[i:i + 4]
                pe = keywds.get('penalize_extend_when_opening', 0)
                keywds['gap_A_fn'] = affine_penalty(openA, extendA, pe)
                keywds['gap_B_fn'] = affine_penalty(openB, extendB, pe)
                i += 4
            else:
                raise ValueError(f'unknown parameter {self.param_names[i]!r}')
        pe = keywds.get('penalize_extend_when_opening', 0)
        default_params = [('match_fn', identity_match(1, 0)), ('gap_A_fn', affine_penalty(0, 0, pe)), ('gap_B_fn', affine_penalty(0, 0, pe)), ('penalize_extend_when_opening', 0), ('penalize_end_gaps', self.align_type == 'global'), ('align_globally', self.align_type == 'global'), ('gap_char', '-'), ('force_generic', 0), ('score_only', 0), ('one_alignment_only', 0)]
        for name, default in default_params:
            keywds[name] = keywds.get(name, default)
        value = keywds['penalize_end_gaps']
        try:
            n = len(value)
        except TypeError:
            keywds['penalize_end_gaps'] = tuple([value] * 2)
        else:
            assert n == 2
        return keywds

    def __call__(self, *args, **keywds):
        """Call the alignment instance already created."""
        keywds = self.decode(*args, **keywds)
        return _align(**keywds)