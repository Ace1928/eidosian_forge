from boto.exception import BotoServerError
class SESLocalAddressCharacterError(SESError):
    """
    An address contained a control or whitespace character.
    """
    pass