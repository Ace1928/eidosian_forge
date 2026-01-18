import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
class InfoArrayValidator(BaseValidator):
    """
    "info_array": {
        "description": "An {array} of plot information.",
        "requiredOpts": [
            "items"
        ],
        "otherOpts": [
            "dflt",
            "freeLength",
            "dimensions"
        ]
    }
    """

    def __init__(self, plotly_name, parent_name, items, free_length=None, dimensions=None, **kwargs):
        super(InfoArrayValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.items = items
        self.dimensions = dimensions if dimensions else 1
        self.free_length = free_length
        self.item_validators = []
        info_array_items = self.items if isinstance(self.items, list) else [self.items]
        for i, item in enumerate(info_array_items):
            element_name = '{name}[{i}]'.format(name=plotly_name, i=i)
            item_validator = InfoArrayValidator.build_validator(item, element_name, parent_name)
            self.item_validators.append(item_validator)

    def description(self):
        desc = "    The '{plotly_name}' property is an info array that may be specified as:".format(plotly_name=self.plotly_name)
        if isinstance(self.items, list):
            if self.dimensions in (1, '1-2'):
                upto = ' up to' if self.free_length and self.dimensions == 1 else ''
                desc += '\n\n    * a list or tuple of{upto} {N} elements where:'.format(upto=upto, N=len(self.item_validators))
                for i, item_validator in enumerate(self.item_validators):
                    el_desc = item_validator.description().strip()
                    desc = desc + '\n({i}) {el_desc}'.format(i=i, el_desc=el_desc)
            if self.dimensions in ('1-2', 2):
                assert self.free_length
                desc += '\n\n    * a 2D list where:'
                for i, item_validator in enumerate(self.item_validators):
                    orig_name = item_validator.plotly_name
                    item_validator.plotly_name = '{name}[i][{i}]'.format(name=self.plotly_name, i=i)
                    el_desc = item_validator.description().strip()
                    desc = desc + '\n({i}) {el_desc}'.format(i=i, el_desc=el_desc)
                    item_validator.plotly_name = orig_name
        else:
            assert self.free_length
            item_validator = self.item_validators[0]
            orig_name = item_validator.plotly_name
            if self.dimensions in (1, '1-2'):
                item_validator.plotly_name = '{name}[i]'.format(name=self.plotly_name)
                el_desc = item_validator.description().strip()
                desc += '\n    * a list of elements where:\n      {el_desc}\n'.format(el_desc=el_desc)
            if self.dimensions in ('1-2', 2):
                item_validator.plotly_name = '{name}[i][j]'.format(name=self.plotly_name)
                el_desc = item_validator.description().strip()
                desc += '\n    * a 2D list where:\n      {el_desc}\n'.format(el_desc=el_desc)
            item_validator.plotly_name = orig_name
        return desc

    @staticmethod
    def build_validator(validator_info, plotly_name, parent_name):
        datatype = validator_info['valType']
        validator_classname = datatype.title().replace('_', '') + 'Validator'
        validator_class = eval(validator_classname)
        kwargs = {k: validator_info[k] for k in validator_info if k not in ['valType', 'description', 'role']}
        return validator_class(plotly_name=plotly_name, parent_name=parent_name, **kwargs)

    def validate_element_with_indexed_name(self, val, validator, inds):
        """
        Helper to add indexes to a validator's name, call validate_coerce on
        a value, then restore the original validator name.

        This makes sure that if a validation error message is raised, the
        property name the user sees includes the index(es) of the offending
        element.

        Parameters
        ----------
        val:
            A value to be validated
        validator
            A validator
        inds
            List of one or more non-negative integers that represent the
            nested index of the value being validated
        Returns
        -------
        val
            validated value

        Raises
        ------
        ValueError
            if val fails validation
        """
        orig_name = validator.plotly_name
        new_name = self.plotly_name
        for i in inds:
            new_name += '[' + str(i) + ']'
        validator.plotly_name = new_name
        try:
            val = validator.validate_coerce(val)
        finally:
            validator.plotly_name = orig_name
        return val

    def validate_coerce(self, v):
        if v is None:
            return None
        elif not is_array(v):
            self.raise_invalid_val(v)
        orig_v = v
        v = to_scalar_or_list(v)
        is_v_2d = v and is_array(v[0])
        if is_v_2d and self.dimensions in ('1-2', 2):
            if is_array(self.items):
                for i, row in enumerate(v):
                    if not is_array(row) or len(row) != len(self.items):
                        self.raise_invalid_val(orig_v[i], [i])
                    for j, validator in enumerate(self.item_validators):
                        row[j] = self.validate_element_with_indexed_name(v[i][j], validator, [i, j])
            else:
                validator = self.item_validators[0]
                for i, row in enumerate(v):
                    if not is_array(row):
                        self.raise_invalid_val(orig_v[i], [i])
                    for j, el in enumerate(row):
                        row[j] = self.validate_element_with_indexed_name(el, validator, [i, j])
        elif v and self.dimensions == 2:
            self.raise_invalid_val(orig_v[0], [0])
        elif not is_array(self.items):
            validator = self.item_validators[0]
            for i, el in enumerate(v):
                v[i] = self.validate_element_with_indexed_name(el, validator, [i])
        elif not self.free_length and len(v) != len(self.item_validators):
            self.raise_invalid_val(orig_v)
        elif self.free_length and len(v) > len(self.item_validators):
            self.raise_invalid_val(orig_v)
        else:
            for i, (el, validator) in enumerate(zip(v, self.item_validators)):
                v[i] = validator.validate_coerce(el)
        return v

    def present(self, v):
        if v is None:
            return None
        elif self.dimensions == 2 or (self.dimensions == '1-2' and v and is_array(v[0])):
            v = copy.deepcopy(v)
            for row in v:
                for i, (el, validator) in enumerate(zip(row, self.item_validators)):
                    row[i] = validator.present(el)
            return tuple((tuple(row) for row in v))
        else:
            v = copy.copy(v)
            for i, (el, validator) in enumerate(zip(v, self.item_validators)):
                v[i] = validator.present(el)
            return tuple(v)