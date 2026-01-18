from collections import OrderedDict, defaultdict
import os
import os.path as op
from pathlib import Path
import shutil
import socket
from copy import deepcopy
from glob import glob
from logging import INFO
from tempfile import mkdtemp
from ... import config, logging
from ...utils.misc import flatten, unflatten, str2bool, dict_diff
from ...utils.filemanip import (
from ...interfaces.base import (
from ...interfaces.base.specs import get_filecopy_info
from .utils import (
from .base import EngineBase
class JoinNode(Node):
    """Wraps interface objects that join inputs into a list.

    Examples
    --------

    >>> import nipype.pipeline.engine as pe
    >>> from nipype import Node, JoinNode, Workflow
    >>> from nipype.interfaces.utility import IdentityInterface
    >>> from nipype.interfaces import (ants, dcm2nii, fsl)
    >>> wf = Workflow(name='preprocess')
    >>> inputspec = Node(IdentityInterface(fields=['image']),
    ...                     name='inputspec')
    >>> inputspec.iterables = [('image',
    ...                        ['img1.nii', 'img2.nii', 'img3.nii'])]
    >>> img2flt = Node(fsl.ImageMaths(out_data_type='float'),
    ...                   name='img2flt')
    >>> wf.connect(inputspec, 'image', img2flt, 'in_file')
    >>> average = JoinNode(ants.AverageImages(), joinsource='inputspec',
    ...                       joinfield='images', name='average')
    >>> wf.connect(img2flt, 'out_file', average, 'images')
    >>> realign = Node(fsl.FLIRT(), name='realign')
    >>> wf.connect(img2flt, 'out_file', realign, 'in_file')
    >>> wf.connect(average, 'output_average_image', realign, 'reference')
    >>> strip = Node(fsl.BET(), name='strip')
    >>> wf.connect(realign, 'out_file', strip, 'in_file')

    """

    def __init__(self, interface, name, joinsource, joinfield=None, unique=False, **kwargs):
        """

        Parameters
        ----------
        interface : interface object
            node specific interface (fsl.Bet(), spm.Coregister())
        name : alphanumeric string
            node specific name
        joinsource : node name
            name of the join predecessor iterable node
        joinfield : string or list of strings
            name(s) of list input fields that will be aggregated.
            The default is all of the join node input fields.
        unique : flag indicating whether to ignore duplicate input values

        See Node docstring for additional keyword arguments.
        """
        super(JoinNode, self).__init__(interface, name, **kwargs)
        self._joinsource = None
        self.joinsource = joinsource
        'the join predecessor iterable node'
        if not joinfield:
            joinfield = self._interface.inputs.copyable_trait_names()
        elif isinstance(joinfield, (str, bytes)):
            joinfield = [joinfield]
        self.joinfield = joinfield
        'the fields to join'
        self._inputs = self._override_join_traits(self._interface.inputs, self.joinfield)
        'the override inputs'
        self._unique = unique
        'flag indicating whether to ignore duplicate input values'
        self._next_slot_index = 0
        'the joinfield index assigned to an iterated input'

    @property
    def joinsource(self):
        return self._joinsource

    @joinsource.setter
    def joinsource(self, value):
        """Set the joinsource property. If the given value is a Node,
        then the joinsource is set to the node name.
        """
        if isinstance(value, Node):
            value = value.name
        self._joinsource = value

    @property
    def inputs(self):
        """The JoinNode inputs include the join field overrides."""
        return self._inputs

    def _add_join_item_fields(self):
        """Add new join item fields assigned to the next iterated
        input

        This method is intended solely for workflow graph expansion.

        Examples
        --------

        >>> from nipype.interfaces.utility import IdentityInterface
        >>> import nipype.pipeline.engine as pe
        >>> from nipype import Node, JoinNode, Workflow
        >>> inputspec = Node(IdentityInterface(fields=['image']),
        ...    name='inputspec'),
        >>> join = JoinNode(IdentityInterface(fields=['images', 'mask']),
        ...    joinsource='inputspec', joinfield='images', name='join')
        >>> join._add_join_item_fields()
        {'images': 'imagesJ1'}

        Return the {base field: slot field} dictionary
        """
        idx = self._next_slot_index
        newfields = dict([(field, self._add_join_item_field(field, idx)) for field in self.joinfield])
        logger.debug('Added the %s join item fields %s.', self, newfields)
        self._next_slot_index += 1
        return newfields

    def _add_join_item_field(self, field, index):
        """Add new join item fields qualified by the given index

        Return the new field name
        """
        name = '%sJ%d' % (field, index + 1)
        trait = self._inputs.trait(field, False, True)
        self._inputs.add_trait(name, trait)
        return name

    def _override_join_traits(self, basetraits, fields):
        """Convert the given join fields to accept an input that
        is a list item rather than a list. Non-join fields
        delegate to the interface traits.

        Return the override DynamicTraitedSpec
        """
        dyntraits = DynamicTraitedSpec()
        if fields is None:
            fields = basetraits.copyable_trait_names()
        else:
            for field in fields:
                if not basetraits.trait(field):
                    raise ValueError('The JoinNode %s does not have a field named %s' % (self.name, field))
        for name, trait in list(basetraits.items()):
            if name in fields and len(trait.inner_traits) == 1:
                item_trait = trait.inner_traits[0]
                dyntraits.add_trait(name, item_trait)
                setattr(dyntraits, name, Undefined)
                logger.debug('Converted the join node %s field %s trait type from %s to %s', self, name, trait.trait_type.info(), item_trait.info())
            else:
                dyntraits.add_trait(name, traits.Any)
                setattr(dyntraits, name, Undefined)
        return dyntraits

    def _run_command(self, execute, copyfiles=True):
        """Collates the join inputs prior to delegating to the superclass."""
        self._collate_join_field_inputs()
        return super(JoinNode, self)._run_command(execute, copyfiles)

    def _collate_join_field_inputs(self):
        """
        Collects each override join item field into the interface join
        field input."""
        for field in self.inputs.copyable_trait_names():
            if field in self.joinfield:
                val = self._collate_input_value(field)
                try:
                    setattr(self._interface.inputs, field, val)
                except Exception as e:
                    raise ValueError('>>JN %s %s %s %s %s: %s' % (self, field, val, self.inputs.copyable_trait_names(), self.joinfield, e))
            elif hasattr(self._interface.inputs, field):
                val = getattr(self._inputs, field)
                if isdefined(val):
                    setattr(self._interface.inputs, field, val)
        logger.debug('Collated %d inputs into the %s node join fields', self._next_slot_index, self)

    def _collate_input_value(self, field):
        """
        Collects the join item field values into a list or set value for
        the given field, as follows:

        - If the field trait is a Set, then the values are collected into
        a set.

        - Otherwise, the values are collected into a list which preserves
        the iterables order. If the ``unique`` flag is set, then duplicate
        values are removed but the iterables order is preserved.
        """
        val = [self._slot_value(field, idx) for idx in range(self._next_slot_index)]
        basetrait = self._interface.inputs.trait(field)
        if isinstance(basetrait.trait_type, traits.Set):
            return set(val)
        if self._unique:
            return list(OrderedDict.fromkeys(val))
        return val

    def _slot_value(self, field, index):
        slot_field = '%sJ%d' % (field, index + 1)
        try:
            return getattr(self._inputs, slot_field)
        except AttributeError as e:
            raise AttributeError('The join node %s does not have a slot field %s to hold the %s value at index %d: %s' % (self, slot_field, field, index, e))