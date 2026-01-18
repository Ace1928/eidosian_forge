import json
import re
from typing import List, Dict

def parse_chat_history(input_file_path: str) -> List[Dict]:
    """
    Parses the chat history from a text file and returns a list of message dictionaries.

    Each message in the text file should follow the format:
    Message [Number]:
    [Author]: [Content]
    Timestamp: [Timestamp]
    Text Content:
    [Text Content]
    Code Content:
    [Code Content]
    Attachments:
    [Attachments]
    ----------------------

    Args:
        input_file_path (str): Path to the input text file containing the chat history.

    Returns:
        List[Dict]: A list of dictionaries, each representing a parsed message.
    """
    messages = []
    current_message = {}
    section = None
    in_code_block = False
    current_code = ""
    current_code_language = "plaintext"

    # Define regex patterns
    message_start_regex = re.compile(r'^Message\s+(\d+):\s*$')
    author_regex = re.compile(r'^([^:]+):\s*(.*)$')
    timestamp_regex = re.compile(r'^Timestamp:\s*(.*)$')
    code_fence_regex = re.compile(r'^```(\w+)?\s*$')
    separator_regex = re.compile(r'^-+$')

    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.rstrip('\n')

            # Detect code fences
            code_fence_match = code_fence_regex.match(line)
            if code_fence_match:
                if not in_code_block:
                    # Start of a code block
                    in_code_block = True
                    current_code_language = code_fence_match.group(1) if code_fence_match.group(1) else "plaintext"
                    current_code = ""
                else:
                    # End of a code block
                    in_code_block = False
                    # Append the code block to code_content
                    if 'code_content' not in current_message:
                        current_message['code_content'] = []
                    current_message['code_content'].append({
                        "language": current_code_language,
                        "code": current_code.strip()
                    })
                    current_code = ""
                    current_code_language = "plaintext"
                continue  # Move to the next line

            if in_code_block:
                current_code += line + "\n"
                continue  # Continue collecting code lines

            # Check for the start of a new message
            message_start_match = message_start_regex.match(line)
            if message_start_match:
                # Save the previous message if exists
                if current_message:
                    messages.append(current_message)
                    current_message = {}
                # Initialize new message
                message_number = int(message_start_match.group(1))
                current_message['original_message_number'] = message_number
                current_message['message_id'] = len(messages) + 1
                current_message['code_content'] = []
                current_message['attachments'] = ""
                section = None
                continue  # Move to the next line

            # Check for Author line
            author_match = author_regex.match(line)
            if author_match and section is None:
                current_message['author'] = author_match.group(1).strip()
                author_content = author_match.group(2).strip()
                if author_content:
                    current_message['text_content'] = author_content + "\n"
                else:
                    current_message['text_content'] = ""
                continue  # Move to the next line

            # Check for Timestamp
            timestamp_match = timestamp_regex.match(line)
            if timestamp_match:
                current_message['timestamp'] = timestamp_match.group(1).strip()
                section = None
                continue  # Move to the next line

            # Check for section headers
            if line.startswith('Text Content:'):
                section = 'text_content'
                if 'text_content' not in current_message:
                    current_message['text_content'] = ""
                continue  # Move to the next line
            elif line.startswith('Code Content:'):
                section = 'code_content'
                # Initialize code_content list if not already
                if 'code_content' not in current_message:
                    current_message['code_content'] = []
                continue  # Move to the next line
            elif line.startswith('Attachments:'):
                section = 'attachments'
                if 'attachments' not in current_message:
                    current_message['attachments'] = ""
                continue  # Move to the next line
            elif separator_regex.match(line):
                section = None
                continue  # Move to the next line

            # Append content based on the current section
            if section == 'text_content':
                if 'text_content' not in current_message:
                    current_message['text_content'] = ""
                current_message['text_content'] += line + "\n"
            elif section == 'attachments':
                current_message['attachments'] += line + "\n"
            # Code Content is handled via code fences

        # After the loop, append the last message if exists
        if current_message:
            # If still inside a code block, close it
            if in_code_block:
                current_message['code_content'].append({
                    "language": current_code_language,
                    "code": current_code.strip()
                })
            messages.append(current_message)

    return messages


def main():
    input_file_path = "C:/Users/ace19/OneDrive/Desktop/Development/Eidos_chat_history.txt"
    output_file_path = "C:/Users/ace19/OneDrive/Desktop/Development/Eidos_Structured_Chat_Messages.json"

    try:
        messages = parse_chat_history(input_file_path)
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(messages, outfile, indent=4, ensure_ascii=False)
        print(f"Successfully parsed {len(messages)} messages and saved to {output_file_path}.")
    except FileNotFoundError:
        print(f"Input file not found at {input_file_path}. Please check the path and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
