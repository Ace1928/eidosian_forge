from math import sqrt
def _get_truncation_coordinates(self, cutlength=0):
    """Count (UI, OI) pairs for truncation points until we find the segment where (ui, oi) crosses the truncation line.

        :param cutlength: Optional parameter to start counting from (ui, oi)
        coordinates gotten by stemming at this length. Useful for speeding up
        the calculations when you know the approximate location of the
        intersection.
        :type cutlength: int
        :return: List of coordinate pairs that define the truncation line
        :rtype: list(tuple(float, float))
        """
    words = get_words_from_dictionary(self.lemmas)
    maxlength = max((len(word) for word in words))
    coords = []
    while cutlength <= maxlength:
        pair = self._get_truncation_indexes(words, cutlength)
        if pair not in coords:
            coords.append(pair)
        if pair == (0.0, 0.0):
            return coords
        if len(coords) >= 2 and pair[0] > 0.0:
            derivative1 = _get_derivative(coords[-2])
            derivative2 = _get_derivative(coords[-1])
            if derivative1 >= self.sw >= derivative2:
                return coords
        cutlength += 1
    return coords