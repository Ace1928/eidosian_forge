from __future__ import annotations
from string import Template
from typing import TYPE_CHECKING
from monty.io import zopen
from pymatgen.io.core import InputGenerator, InputSet
class TemplateInputGen(InputGenerator):
    """
    Concrete implementation of InputGenerator that is based on a single template input
    file with variables.

    This class is provided as a low-barrier way to support new codes and to provide
    an intuitive way for users to transition from manual scripts to pymatgen I/O
    classes.
    """

    def get_input_set(self, template: str | Path, variables: dict | None=None, filename: str='input.txt'):
        """
        Args:
            template: the input file template containing variable strings to be
                replaced.
            variables: dict of variables to replace in the template. Keys are the
                text to replaced with the values, e.g. {"TEMPERATURE": 298} will
                replace the text $TEMPERATURE in the template. See Python's
                Template.safe_substitute() method documentation for more details.
            filename: name of the file to be written.
        """
        self.template = template
        self.variables = variables or {}
        self.filename = filename
        with zopen(self.template, mode='r') as file:
            template_str = file.read()
        self.data = Template(template_str).safe_substitute(**self.variables)
        return InputSet({self.filename: self.data})