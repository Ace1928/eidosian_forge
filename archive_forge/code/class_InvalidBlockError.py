from gradio_client.documentation import document
class InvalidBlockError(ValueError):
    """Raised when an event in a Blocks contains a reference to a Block that is not in the original Blocks"""
    pass