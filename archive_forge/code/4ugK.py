import os
import re
from typing import List, Tuple


class FileProcessor:
    """
    Handles the extraction of text from various document types within a specified folder.
    Supported file types include .txt, .md, .docx, .pdf. Script and program files are excluded.
    This class is designed for easy maintenance and scalability through modular function design.
    """

    def __init__(self, folder_path: str):
        """
        Initializes the FileProcessor with the path to the directory containing the documents to be processed.
        :param folder_path: str - Path to the directory containing the files.
        """
        self.folder_path = folder_path

    def _list_files(self) -> List[str]:
        """
        Lists all eligible files in the specified directory that match supported document types,
        excluding unsupported file types and hidden files.
        :return: List[str] - A list of file paths that are eligible for processing.
        """
        supported_extensions = (".txt", ".md", ".docx", ".pdf")
        return [
            os.path.join(self.folder_path, file)
            for file in os.listdir(self.folder_path)
            if file.endswith(supported_extensions) and not file.startswith(".")
        ]

    def _read_text_file(self, file_path: str) -> str:
        """
        Reads and returns the content of a text file.
        :param file_path: str - Path to the text file.
        :return: str - The content of the file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _read_markdown_file(self, file_path: str) -> str:
        """
        Reads and returns the content of a markdown file.
        :param file_path: str - Path to the markdown file.
        :return: str - The content of the file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _read_docx_file(self, file_path: str) -> str:
        """
        Reads and extracts text from a DOCX file.
        :param file_path: str - Path to the DOCX file.
        :return: str - The extracted text.
        """
        from docx import Document

        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    def _read_pdf_file(self, file_path: str) -> str:
        """
        Reads and extracts text from a PDF file.
        :param file_path: str - Path to the PDF file.
        :return: str - The extracted text.
        """
        import PyPDF2

        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join(
                page.extract_text()
                for page in pdf_reader.pages
                if page.extract_text() is not None
            )

    def process_files(self) -> List[Tuple[str, str]]:
        """
        Processes all documents in the specified folder, extracting text and returning a list of text data.
        Utilizes specific reader functions for different file types to optimize text extraction.
        :return: List[Tuple[str, str]] - A list of tuples containing file paths and their extracted text.
        """
        files = self._list_files()
        extracted_data = []
        for file_path in files:
            if file_path.endswith(".txt") or file_path.endswith(".md"):
                text = self._read_text_file(file_path)
            elif file_path.endswith(".docx"):
                text = self._read_docx_file(file_path)
            elif file_path.endswith(".pdf"):
                text = self._read_pdf_file(file_path)
            else:
                continue  # Skip unsupported file types
            extracted_data.append((file_path, text))
        return extracted_data
