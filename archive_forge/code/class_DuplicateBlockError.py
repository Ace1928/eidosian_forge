from gradio_client.documentation import document
class DuplicateBlockError(ValueError):
    """Raised when a Blocks contains more than one Block with the same id"""
    pass