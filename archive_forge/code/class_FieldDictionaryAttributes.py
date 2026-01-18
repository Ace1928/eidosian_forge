from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class FieldDictionaryAttributes:
    """TABLE 8.69 Entries common to all field dictionaries (PDF 1.7 reference)."""
    FT = '/FT'
    Parent = '/Parent'
    Kids = '/Kids'
    T = '/T'
    TU = '/TU'
    TM = '/TM'
    Ff = '/Ff'
    V = '/V'
    DV = '/DV'
    AA = '/AA'
    Opt = '/Opt'

    class FfBits:
        ReadOnly = 1 << 0
        Required = 1 << 1
        NoExport = 1 << 2
        Multiline = 1 << 12
        Password = 1 << 13
        NoToggleToOff = 1 << 14
        Radio = 1 << 15
        Pushbutton = 1 << 16
        Combo = 1 << 17
        Edit = 1 << 18
        Sort = 1 << 19
        FileSelect = 1 << 20
        MultiSelect = 1 << 21
        DoNotSpellCheck = 1 << 22
        DoNotScroll = 1 << 23
        Comb = 1 << 24
        RadiosInUnison = 1 << 25
        RichText = 1 << 25
        CommitOnSelChange = 1 << 26

    @classmethod
    def attributes(cls) -> Tuple[str, ...]:
        """
        Get a tuple of all the attributes present in a Field Dictionary.

        This method returns a tuple of all the attribute constants defined in
        the FieldDictionaryAttributes class. These attributes correspond to the
        entries that are common to all field dictionaries as specified in the
        PDF 1.7 reference.

        Returns:
            A tuple containing all the attribute constants.
        """
        return (cls.TM, cls.T, cls.FT, cls.Parent, cls.TU, cls.Ff, cls.V, cls.DV, cls.Kids, cls.AA)

    @classmethod
    def attributes_dict(cls) -> Dict[str, str]:
        """
        Get a dictionary of attribute keys and their human-readable names.

        This method returns a dictionary where the keys are the attribute
        constants defined in the FieldDictionaryAttributes class and the values
        are their corresponding human-readable names. These attributes
        correspond to the entries that are common to all field dictionaries as
        specified in the PDF 1.7 reference.

        Returns:
            A dictionary containing attribute keys and their names.
        """
        return {cls.FT: 'Field Type', cls.Parent: 'Parent', cls.T: 'Field Name', cls.TU: 'Alternate Field Name', cls.TM: 'Mapping Name', cls.Ff: 'Field Flags', cls.V: 'Value', cls.DV: 'Default Value'}