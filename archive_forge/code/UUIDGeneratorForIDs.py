
import uuid

def generate_unique_identifier():
    return str(uuid.uuid4())

# Example usage
if __name__ == "__main__":
    unique_identifier = generate_unique_identifier()
    print(f"Generated Unique Identifier: {unique_identifier}")
