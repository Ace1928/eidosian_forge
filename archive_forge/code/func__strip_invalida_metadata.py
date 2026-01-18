from __future__ import annotations
import json
import pprint
import warnings
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional
from ._imports import import_item
from .corpus.words import generate_corpus_id
from .json_compat import ValidationError, _validator_for_name, get_current_validator
from .reader import get_version
from .warnings import DuplicateCellId, MissingIDFieldWarning
def _strip_invalida_metadata(nbdict: Any, version: int, version_minor: int, relax_add_props: bool) -> int:
    """
    This function tries to extract metadata errors from the validator and fix
    them if necessary. This mostly mean stripping unknown keys from metadata
    fields, or removing metadata fields altogether.

    Parameters
    ----------
    nbdict : dict
        notebook document
    version : int
    version_minor : int
    relax_add_props : bool
        Whether to allow extra property in the Json schema validating the
        notebook.

    Returns
    -------
    int
        number of modifications

    """
    errors = _get_errors(nbdict, version, version_minor, relax_add_props)
    changes = 0
    if len(list(errors)) > 0:
        validator = get_validator(version=version, version_minor=version_minor, relax_add_props=relax_add_props, name='jsonschema')
        if not validator:
            msg = f'No jsonschema for validating v{version}.{version_minor} notebooks'
            raise ValidationError(msg)
        errors = validator.iter_errors(nbdict)
        error_tree = validator.error_tree(errors)
        if 'metadata' in error_tree:
            for key in error_tree['metadata']:
                nbdict['metadata'].pop(key, None)
                changes += 1
        if 'cells' in error_tree:
            number_of_cells = len(nbdict.get('cells', 0))
            for cell_idx in range(number_of_cells):
                if 'oneOf' in error_tree['cells'][cell_idx].errors:
                    intended_cell_type = nbdict['cells'][cell_idx]['cell_type']
                    schemas_by_index = [ref['$ref'] for ref in error_tree['cells'][cell_idx].errors['oneOf'].schema['oneOf']]
                    cell_type_definition_name = f'#/definitions/{intended_cell_type}_cell'
                    if cell_type_definition_name in schemas_by_index:
                        schema_index = schemas_by_index.index(cell_type_definition_name)
                        for error in error_tree['cells'][cell_idx].errors['oneOf'].context:
                            rel_path = error.relative_path
                            error_for_intended_schema = error.schema_path[0] == schema_index
                            is_top_level_metadata_key = len(rel_path) == 2 and rel_path[0] == 'metadata'
                            if error_for_intended_schema and is_top_level_metadata_key:
                                nbdict['cells'][cell_idx]['metadata'].pop(rel_path[1], None)
                                changes += 1
    return changes