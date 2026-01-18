class IDeclarative(Interface):

    def __init__(**kw):
        """
        Instantiates this class with all the keywords being used to
        update the instance variables.
        """

    def __call__(**kw):
        """
        Returns a copy with all attributes using the given keywords,
        being updated.
        """