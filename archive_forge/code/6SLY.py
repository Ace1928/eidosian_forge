import os
import re
from typing import List, Tuple, Dict, Optional
import logging
from docx import Document
import PyPDF2


class FileProcessor:
    """
    Manages the extraction of textual content from various document types within a designated folder.
    Supported file types include .txt, .md, .docx, .pdf. Script and program files are systematically excluded.
    This class is architected for optimal maintenance and scalability through a modular function design.
    """

    def __init__(self, folder_path: str):
        """
        Constructs the FileProcessor with the path to the directory containing the documents to be processed.
        :param folder_path: str - Path to the directory housing the files.
        """
        self.folder_path = folder_path
        logging.basicConfig(level=logging.INFO)

    def _list_files(self) -> List[str]:
        """
        Enumerates all eligible files in the specified directory that conform to supported document types,
        methodically excluding unsupported file types and hidden files.
        :return: List[str] - A list of file paths that are eligible for processing.
        """
        supported_extensions = (".txt", ".md", ".docx", ".pdf")
        eligible_files = [
            os.path.join(self.folder_path, file)
            for file in os.listdir(self.folder_path)
            if file.endswith(supported_extensions) and not file.startswith(".")
        ]
        logging.info(f"Eligible files listed: {eligible_files}")
        return eligible_files

    def _read_text_file(self, file_path: str) -> str:
        """
        Retrieves and returns the content of a text file.
        :param file_path: str - Path to the text file.
        :return: str - The content of the file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        logging.info(f"Text file read from {file_path}")
        return content

    def _read_markdown_file(self, file_path: str) -> str:
        """
        Retrieves and returns the content of a markdown file.
        :param file_path: str - Path to the markdown file.
        :return: str - The content of the file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        logging.info(f"Markdown file read from {file_path}")
        return content

    def _read_docx_file(self, file_path: str) -> str:
        """
        Extracts and returns text from a DOCX file.
        :param file_path: str - Path to the DOCX file.
        :return: str - The extracted text.
        """
        doc = Document(file_path)
        extracted_text = "\n".join(para.text for para in doc.paragraphs)
        logging.info(f"DOCX file read from {file_path}")
        return extracted_text

    def _read_pdf_file(self, file_path: str) -> str:
        """
        Extracts and returns text from a PDF file.
        :param file_path: str - Path to the PDF file.
        :return: str - The extracted text.
        """
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            extracted_text = "\n".join(
                page.extract_text()
                for page in pdf_reader.pages
                if page.extract_text() is not None
            )
        logging.info(f"PDF file read from {file_path}")
        return extracted_text

    def process_files(self) -> List[Tuple[str, str]]:
        """
        Processes all documents in the specified folder, extracting text and returning a list of text data.
        Employs specific reader functions for different file types to optimize text extraction.
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
                logging.warning(f"Unsupported file type encountered: {file_path}")
                continue  # Skip unsupported file types
            extracted_data.append((file_path, text))
            logging.info(f"File processed: {file_path}")
        return extracted_data
