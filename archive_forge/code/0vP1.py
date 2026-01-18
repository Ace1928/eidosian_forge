import os
import json
import csv
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union, Optional, Tuple, List
import pandas as pd
import logging
import yaml
import pickle
import configparser
import markdown
import docx
import openpyxl
import sqlite3
import PyPDF2
import PIL.Image

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class UniversalDataReader:
    """
    A class meticulously designed to read and process various file types through a universal interface.
    This class encapsulates methods that identify and process file content with precision and adaptability,
    ensuring optimal performance and extensibility.
    """

    def __init__(self, file_path: str):
        """
        Initialize the UniversalDataReader with a specific file path.

        :param file_path: The path to the file to be read.
        :type file_path: str
        """
        self.file_path: str = file_path
        self.file_type: str = self.identify_file_type()
        logging.info(
            f"UniversalDataReader initialized for file: {file_path} with identified type: {self.file_type}"
        )

    def identify_file_type(self) -> str:
        """
        Determine the file type by extracting and analyzing the file extension.

        :return: A string representing the file type.
        :rtype: str
        """
        _, file_extension = os.path.splitext(self.file_path)
        file_extension = file_extension.lower()
        logging.debug(f"File extension identified: {file_extension}")
        return file_extension

    def read_file(
        self,
    ) -> Union[
        str,
        Dict[str, Any],
        pd.DataFrame,
        ET.ElementTree,
        configparser.ConfigParser,
        object,
        List[str],
        docx.document.Document,
        openpyxl.workbook.workbook.Workbook,
        sqlite3.Connection,
        PyPDF2.PdfReader,
        PIL.Image.Image,
    ]:
        """
        Read the file based on its type and return its content in an appropriate format.

        :return: The content of the file, formatted according to its type.
        :rtype: Union[str, Dict[str, Any], pd.DataFrame, ET.ElementTree, configparser.ConfigParser, object, List[str], docx.document.Document, openpyxl.workbook.workbook.Workbook, sqlite3.Connection, PyPDF2.PdfReader, PIL.Image.Image]
        """
        try:
            if self.file_type == ".json":
                return self.read_json()
            elif self.file_type == ".csv":
                return self.read_csv()
            elif self.file_type == ".xml":
                return self.read_xml()
            elif self.file_type in [".txt", ".log"]:
                return self.read_text()
            elif self.file_type == ".yaml" or self.file_type == ".yml":
                return self.read_yaml()
            elif self.file_type == ".ini":
                return self.read_ini()
            elif self.file_type == ".pkl":
                return self.read_pickle()
            elif self.file_type == ".md":
                return self.read_markdown()
            elif self.file_type == ".docx":
                return self.read_docx()
            elif self.file_type == ".xlsx":
                return self.read_excel()
            elif self.file_type == ".db":
                return self.read_database()
            elif self.file_type == ".pdf":
                return self.read_pdf()
            elif self.file_type in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                return self.read_image()
            else:
                logging.error(f"Unsupported file type: {self.file_type}")
                raise ValueError(f"Unsupported file type: {self.file_type}")
        except Exception as e:
            logging.error(f"Error reading file {self.file_path}: {str(e)}")
            raise

    def read_markdown(self) -> List[str]:
        """
        Read a Markdown file and return its content as a list of strings, each representing a line.

        :return: The content of the Markdown file.
        :rtype: List[str]
        """
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = file.readlines()
            logging.debug(f"Markdown data read successfully from {self.file_path}")
            return data

    def read_docx(self) -> docx.document.Document:
        """
        Read a DOCX file and return its content as a docx Document object.

        :return: The content of the DOCX file.
        :rtype: docx.document.Document
        """
        doc = docx.Document(self.file_path)
        logging.debug(f"DOCX data read successfully from {self.file_path}")
        return doc

    def read_excel(self) -> openpyxl.workbook.workbook.Workbook:
        """
        Read an Excel file and return its content as an openpyxl Workbook object.

        :return: The content of the Excel file.
        :rtype: openpyxl.workbook.workbook.Workbook
        """
        workbook = openpyxl.load_workbook(self.file_path)
        logging.debug(f"Excel data read successfully from {self.file_path}")
        return workbook

    def read_database(self) -> sqlite3.Connection:
        """
        Read a database file and return its connection object.

        :return: The connection to the database.
        :rtype: sqlite3.Connection
        """
        connection = sqlite3.connect(self.file_path)
        logging.debug(
            f"Database connection established successfully from {self.file_path}"
        )
        return connection

    def read_pdf(self) -> PyPDF2.PdfReader:
        """
        Read a PDF file and return its content as a PyPDF2 PdfReader object.

        :return: The content of the PDF file.
        :rtype: PyPDF2.PdfReader
        """
        with open(self.file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            logging.debug(f"PDF data read successfully from {self.file_path}")
            return reader

    def read_image(self) -> PIL.Image.Image:
        """
        Read an image file and return its content as a PIL Image object.

        :return: The content of the image file.
        :rtype: PIL.Image.Image
        """
        image = PIL.Image.open(self.file_path)
        logging.debug(f"Image data read successfully from {self.file_path}")
        return image

    def read_json(self) -> Dict[str, Any]:
        """
        Read a JSON file and return its content as a dictionary.

        :return: The content of the JSON file parsed into a dictionary.
        :rtype: Dict[str, Any]
        """
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            logging.debug(f"JSON data read successfully from {self.file_path}")
            return data

    def read_csv(self) -> pd.DataFrame:
        """
        Read a CSV file and return its content as a pandas DataFrame.

        :return: The content of the CSV file parsed into a DataFrame.
        :rtype: pd.DataFrame
        """
        data = pd.read_csv(self.file_path)
        logging.debug(f"CSV data read successfully from {self.file_path}")
        return data

    def read_xml(self) -> ET.ElementTree:
        """
        Read an XML file and return its content as an ElementTree object.

        :return: The content of the XML file parsed into an ElementTree.
        :rtype: ET.ElementTree
        """
        tree = ET.parse(self.file_path)
        logging.debug(f"XML data read successfully from {self.file_path}")
        return tree

    def read_text(self) -> str:
        """
        Read a text file and return its content as a string.

        :return: The content of the text file.
        :rtype: str
        """
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = file.read()
            logging.debug(f"Text data read successfully from {self.file_path}")
            return data

    def read_yaml(self) -> Dict[str, Any]:
        """
        Read a YAML file and return its content as a dictionary.

        :return: The content of the YAML file parsed into a dictionary.
        :rtype: Dict[str, Any]
        """
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            logging.debug(f"YAML data read successfully from {self.file_path}")
            return data

    def read_ini(self) -> configparser.ConfigParser:
        """
        Read an INI file and return its content as a ConfigParser object.

        :return: The content of the INI file parsed into a ConfigParser.
        :rtype: configparser.ConfigParser
        """
        config = configparser.ConfigParser()
        config.read(self.file_path)
        logging.debug(f"INI data read successfully from {self.file_path}")
        return config

    def read_pickle(self) -> object:
        """
        Read a pickle file and return its content as a Python object.

        :return: The content of the pickle file deserialized into a Python object.
        :rtype: object
        """
        with open(self.file_path, "rb") as file:
            data = pickle.load(file)
            logging.debug(f"Pickle data read successfully from {self.file_path}")
            return data


# Example usage:
# data_reader = UniversalDataReader('path_to_your_file.extension')
# content = data_reader.read_file()
# print(content)
