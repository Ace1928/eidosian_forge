import doctest
import re
import textwrap
import numpy as np
class TfDoctestOutputChecker(doctest.OutputChecker, object):
    """Customizes how `want` and `got` are compared, see `check_output`."""

    def __init__(self, *args, **kwargs):
        super(TfDoctestOutputChecker, self).__init__(*args, **kwargs)
        self.extract_floats = _FloatExtractor()
        self.text_good = None
        self.float_size_good = None
    _ADDRESS_RE = re.compile('\\bat 0x[0-9a-f]*?>')
    _NUMPY_OUTPUT_RE = re.compile('<tf.Tensor.*?numpy=(.*?)>', re.DOTALL)

    def _allclose(self, want, got, rtol=0.001, atol=0.001):
        return np.allclose(want, got, rtol=rtol, atol=atol)

    def _tf_tensor_numpy_output(self, string):
        modified_string = self._NUMPY_OUTPUT_RE.sub('\\1', string)
        return (modified_string, modified_string != string)
    MESSAGE = textwrap.dedent('\n\n        #############################################################\n        Check the documentation (https://www.tensorflow.org/community/contribute/docs_ref) on how to\n        write testable docstrings.\n        #############################################################')

    def check_output(self, want, got, optionflags):
        """Compares the docstring output to the output gotten by running the code.

    Python addresses in the output are replaced with wildcards.

    Float values in the output compared as using `np.allclose`:

      * Float values are extracted from the text and replaced with wildcards.
      * The wildcard text is compared to the actual output.
      * The float values are compared using `np.allclose`.

    The method returns `True` if both the text comparison and the numeric
    comparison are successful.

    The numeric comparison will fail if either:

      * The wrong number of floats are found.
      * The float values are not within tolerence.

    Args:
      want: The output in the docstring.
      got: The output generated after running the snippet.
      optionflags: Flags passed to the doctest.

    Returns:
      A bool, indicating if the check was successful or not.
    """
        if got and (not want):
            return True
        if want is None:
            want = ''
        if want == got:
            return True
        want = self._ADDRESS_RE.sub('at ...>', want)
        want, want_changed = self._tf_tensor_numpy_output(want)
        if want_changed:
            got, _ = self._tf_tensor_numpy_output(got)
        want_text_parts, self.want_floats = self.extract_floats(want)
        want_text_parts = [part.strip(' ') for part in want_text_parts]
        want_text_wild = '...'.join(want_text_parts)
        if '....' in want_text_wild:
            want_text_wild = re.sub('\\.\\.\\.\\.+', '...', want_text_wild)
        _, self.got_floats = self.extract_floats(got)
        self.text_good = super(TfDoctestOutputChecker, self).check_output(want=want_text_wild, got=got, optionflags=optionflags)
        if not self.text_good:
            return False
        if self.want_floats.size == 0:
            return True
        self.float_size_good = self.want_floats.size == self.got_floats.size
        if self.float_size_good:
            return self._allclose(self.want_floats, self.got_floats)
        else:
            return False

    def output_difference(self, example, got, optionflags):
        got = [got]
        if self.text_good:
            if not self.float_size_good:
                got.append('\n\nCAUTION: tf_doctest doesn\'t work if *some* of the *float output* is hidden with a "...".')
        got.append(self.MESSAGE)
        got = '\n'.join(got)
        return super(TfDoctestOutputChecker, self).output_difference(example, got, optionflags)