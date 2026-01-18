import copy
import os
Insert a shim target that forces the linker to use 4KB pagesize PDBs.

  This is a workaround for targets with PDBs greater than 1GB in size, the
  limit for the 1KB pagesize PDBs created by the linker by default.

  Arguments:
    target_list: List of target pairs: 'base/base.gyp:base'.
    target_dicts: Dict of target properties keyed on target pair.
    vars: A dictionary of common GYP variables with generator-specific values.
  Returns:
    Tuple of the shimmed version of the inputs.
  