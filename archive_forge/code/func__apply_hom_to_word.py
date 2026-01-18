from . import matrix
def _apply_hom_to_word(word, G):
    if G is None:
        return word
    if hasattr(G, 'generators_in_originals'):
        result = ''
        imgs = G.generators_in_originals()
        for letter in word:
            if letter.isupper():
                result += imgs[ord(letter) - ord('A')].swapcase()[::-1]
            else:
                result += imgs[ord(letter) - ord('a')]
        return result
    raise Exception('Given argument is not a SnapPy fundamental group')