import re
def _expand_ordinal(self, num: str) -> str:
    """
        This method is used to expand ordinals such as '1st', '2nd' into spoken words.
        """
    ordinal_suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
    num = int(num.group(0)[:-2])
    if 10 <= num % 100 and num % 100 <= 20:
        suffix = 'th'
    else:
        suffix = ordinal_suffixes.get(num % 10, 'th')
    return self.number_to_words(num) + suffix