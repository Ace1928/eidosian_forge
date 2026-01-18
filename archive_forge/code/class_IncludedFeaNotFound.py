class IncludedFeaNotFound(FeatureLibError):

    def __str__(self):
        assert self.location is not None
        message = f'The following feature file should be included but cannot be found: {Exception.__str__(self)}'
        return f'{self.location}: {message}'