from gradio_client.documentation import document
class GradioVersionIncompatibleError(Exception):
    """Raised when loading a 3.x space with 4.0"""
    pass