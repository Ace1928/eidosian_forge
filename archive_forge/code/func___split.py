import suds
def __split(self, url):
    """
        Split the given URL into its I{protocol} & I{location} components.

        @param url: A URL.
        @param url: str
        @return: (I{protocol}, I{location})
        @rtype: (str, str)

        """
    parts = url.split('://', 1)
    if len(parts) == 2:
        return parts
    return (None, url)