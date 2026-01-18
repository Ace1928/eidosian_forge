import json
import os
import logging
import argparse
from typing import List, Dict, Optional, Any, Union


def setup_logger(enable_console_logging: bool = True) -> None:
    """
    Configures the logging system, allowing for console logging to be toggled.

    :param enable_console_logging: If True, enables logging to the console.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="md_to_jsonl_conversion.log",
        filemode="a",
    )
    if enable_console_logging:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)


def parse_md_content(file_content: str) -> List[Dict[str, str]]:
    """
    Parses Markdown content to extract conversations.

    :param file_content: The content of the Markdown file as a string.
    :return: A list of dictionaries containing the conversations.
    """
    logger = logging.getLogger("parse_md_content")
    conversations = []
    user_input, assistant_response = "", ""
    capture_mode: Optional[str] = None

    for line in file_content.split("\n"):
        if line.startswith("## USER"):
            if assistant_response:
                conversations.append(
                    {"input": user_input.strip(), "output": assistant_response.strip()}
                )
                user_input, assistant_response = "", ""
            capture_mode = "user"
        elif line.startswith("## ASSISTANT"):
            capture_mode = "assistant"
        elif capture_mode == "user":
            user_input += line.strip() + " "
        elif capture_mode == "assistant":
            assistant_response += line.strip() + " "

    if user_input and assistant_response:
        conversations.append(
            {"input": user_input.strip(), "output": assistant_response.strip()}
        )

    conversations = [conv for conv in conversations if conv["input"] and conv["output"]]
    logger.info(f"Extracted {len(conversations)} conversations from Markdown content.")
    return conversations


def convert_conversations_to_jsonl(
    conversations: List[Dict[str, str]], output_path: str
) -> None:
    """
    Converts conversations to JSONL format and writes them to a file.

    :param conversations: A list of conversation dictionaries.
    :param output_path: The path to the output JSONL file.
    """
    logger = logging.getLogger("convert_to_jsonl")
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")
        logger.info(f"Conversations successfully written to {output_path}")
    except FileNotFoundError as e:
        logger.error(
            f"Failed to write conversations to JSONL due to {e}", exc_info=True
        )


def process_md_file(md_path: str, output_dir: str) -> None:
    """
    Processes a single Markdown file, converting its content to JSONL format.

    :param md_path: The path to the Markdown file.
    :param output_dir: The directory where the JSONL file will be saved.
    """
    logger = logging.getLogger("process_md_file")
    output_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(md_path))[0] + ".jsonl"
    )

    try:
        with open(md_path, "r", encoding="utf-8") as md_file:
            file_content = md_file.read()
            conversations = parse_md_content(file_content)
            convert_conversations_to_jsonl(conversations, output_path)
    except FileNotFoundError as e:
        logger.error(f"Markdown file not found: {e}", exc_info=True)


def process_md_directory(input_dir: str, output_dir: str) -> None:
    """
    Processes all Markdown files in a directory, converting them to JSONL format.

    :param input_dir: The directory containing Markdown files.
    :param output_dir: The directory where JSONL files will be saved.
    """
    logger = logging.getLogger("process_md_directory")
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".md"):
                md_path = os.path.join(root, file)
                process_md_file(md_path, output_dir)

    logger.info(f"All Markdown files in {input_dir} have been processed.")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    :return: An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert Markdown files to JSONL format."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing Markdown files."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save JSONL files."
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logger()
    args = parse_arguments()
    input_directory = args.input_dir
    output_directory = args.output_dir

    try:
        process_md_directory(input_directory, output_directory)
        logger = logging.getLogger("main")
        logger.info("Markdown conversion process completed successfully.")
    except Exception as e:
        logger.critical(
            "An unexpected error occurred during the Markdown conversion process.",
            exc_info=True,
        )
