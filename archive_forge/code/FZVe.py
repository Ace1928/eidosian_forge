class FileProcessor:
    """
    Handles the extraction of text from various document types within a specified folder.
    Supported file types include .txt, .md, .docx, .pdf. Ignores script and program files.
    """

    def __init__(self, folder_path: str):
        """
        Initializes the FileProcessor with the path to the folder containing documents.
        """

    def process_files(self) -> list:
        """
        Processes all documents in the specified folder, extracting text and returning a list of text data.
        Returns a list of tuples containing file paths and their extracted text.
        """
