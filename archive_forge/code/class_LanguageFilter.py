from enum import Enum, Flag
class LanguageFilter(Flag):
    """
    This enum represents the different language filters we can apply to a
    ``UniversalDetector``.
    """
    NONE = 0
    CHINESE_SIMPLIFIED = 1
    CHINESE_TRADITIONAL = 2
    JAPANESE = 4
    KOREAN = 8
    NON_CJK = 16
    ALL = 31
    CHINESE = CHINESE_SIMPLIFIED | CHINESE_TRADITIONAL
    CJK = CHINESE | JAPANESE | KOREAN